"""Detect unwanted robot self-contact while allowing manipulation contacts."""

from .schema import JOINTS


class SelfCollisionError(RuntimeError):
    """A demonstration cannot start or be kept after robot self-contact."""


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
            return {
                "bodies": [self.model.body(body).name for body in bodies],
                "geoms": [self.model.geom(g).name or f"geom_{g}" for g in geoms],
                "penetration_m": float(max(0.0, -contact.dist)),
                "sim_time": float(data.time),
            }
        return None


def collision_message(contact):
    return "Self-collision: " + " touches ".join(contact["bodies"])
