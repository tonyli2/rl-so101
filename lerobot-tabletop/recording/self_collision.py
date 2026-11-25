"""Check self-contact and recording obstacles; allow normal manipulation."""

from .schema import JOINTS


class SelfCollisionError(RuntimeError):
    """A demonstration cannot start or be kept after prohibited robot contact."""


class SelfCollisionChecker:
    """Check MuJoCo's actual contacts between nonadjacent bodies of the arm.

    Robot membership follows the body tree containing the shoulder-pan joint,
    including its fixed base and gripper descendants. World/table/object contact
    is not robot self-contact. Direct parent/child bodies are excluded because
    adjoining links may touch at their mechanical joint.
    """

    def __init__(self, model):
        self.model = model
        root = int(model.jnt_bodyid[model.joint(JOINTS[0]).id])
        while int(model.body_parentid[root]) != 0:
            root = int(model.body_parentid[root])
        self.robot_bodies = set()
        for body in range(1, model.nbody):
            ancestor = body
            while ancestor and ancestor != root:
                ancestor = int(model.body_parentid[ancestor])
            if ancestor == root:
                self.robot_bodies.add(body)

    def _contact_details(self, data, contact, geoms, bodies):
        """Shared diagnostic fields for self-contact and recording obstacles."""
        return {
            "bodies": [self.model.body(body).name for body in bodies],
            "geoms": [self.model.geom(g).name or f"geom_{g}" for g in geoms],
            "penetration_m": float(max(0.0, -contact.dist)),
            "sim_time": float(data.time),
        }

    def check(self, data):
        for contact in data.contact:
            # Contacts within an optional positive margin are proximity, not a
            # touching/intersecting self-collision. Zero distance counts as touch.
            if contact.dist > 0:
                continue
            geoms = [int(g) for g in contact.geom]
            bodies = [int(self.model.geom_bodyid[g]) for g in geoms]
            a, b = bodies
            if a not in self.robot_bodies or b not in self.robot_bodies or a == b:
                continue
            if (
                int(self.model.body_parentid[a]) == b
                or int(self.model.body_parentid[b]) == a
            ):
                continue
            return self._contact_details(data, contact, geoms, bodies)
        return None


def collision_message(contact):
    if contact.get("kind") == "curtain":
        prefix = f"Curtain collision ({contact['curtain']}): "
    elif contact.get("kind") == "upright":
        prefix = "Upright collision: "
    else:
        prefix = "Self-collision: "
    return prefix + " touches ".join(contact["bodies"])


class RecordingCollisionChecker(SelfCollisionChecker):
    """Recording gate for self-contact, upright contact, and paper curtains.

    The historical self-only checker is unchanged. Task-object contacts and
    the visual-only stand bases are not rejected by this gate.
    """

    def __init__(self, model):
        super().__init__(model)
        try:
            self.upright_geom = model.geom("overhead_upright_collision").id
        except KeyError:
            self.upright_geom = None
        self.curtain_geoms = {
            gid: model.geom(gid).name.removeprefix("paper_curtain_")
            for gid in range(model.ngeom)
            if (model.geom(gid).name or "").startswith("paper_curtain_")
        }
        self.obstacle_geoms = set(self.curtain_geoms)
        if self.upright_geom is not None:
            self.obstacle_geoms.add(self.upright_geom)

    def check(self, data):
        self_contact = super().check(data)
        if self_contact is not None or not self.obstacle_geoms:
            return self_contact
        for contact in data.contact:
            if contact.dist > 0:
                continue
            geoms = list(map(int, contact.geom))
            obstacles = self.obstacle_geoms.intersection(geoms)
            if not obstacles:
                continue
            bodies = [int(self.model.geom_bodyid[g]) for g in geoms]
            if not any(body in self.robot_bodies for body in bodies):
                continue
            obstacle = min(obstacles)
            result = self._contact_details(data, contact, geoms, bodies)
            result["kind"] = "curtain" if obstacle in self.curtain_geoms else "upright"
            if obstacle in self.curtain_geoms:
                result["curtain"] = self.curtain_geoms[obstacle]
            return result
        return None
