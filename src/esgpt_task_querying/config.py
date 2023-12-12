"""TODO(justin): Add a module docstring."""

import re
from datetime import timedelta

import ruamel.yaml
from bigtree import Node, preorder_iter


class DotAccessibleDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotAccessibleDict(value)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        else:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")


def load_config(config_path: str) -> DotAccessibleDict:
    yaml = ruamel.yaml.YAML()
    with open(config_path) as file:
        config_dict = yaml.load(file)
    return DotAccessibleDict(config_dict)


def parse_timedelta(time_str):
    if not time_str:
        return timedelta(days=0)

    units = {"days": 0, "hours": 0, "minutes": 0, "seconds": 0}
    pattern = r"(\d+)\s*(seconds|minutes|hours|days)"
    matches = re.findall(pattern, time_str.lower())
    for value, unit in matches:
        units[unit] = int(value)

    return timedelta(
        days=units["days"],
        hours=units["hours"],
        minutes=units["minutes"],
        seconds=units["seconds"],
    )


def build_tree_from_config(cfg):
    nodes = {}
    windows = [x for x, y in cfg.windows.items()]

    for window_name in windows:
        node = Node(window_name)
        window_info = cfg.windows[window_name]

        if window_info.duration:
            if "-" in window_info.duration:
                end_event = -parse_timedelta(window_info.duration)
            else:
                end_event = parse_timedelta(window_info.duration)
        else:
            end_event = f"is_{window_info.end}"

        st_inclusive = False
        end_inclusive = True
        if "st_inclusive" in window_info:
            st_inclusive = window_info.st_inclusive
        if "end_inclusive" in window_info:
            end_inclusive = window_info.end_inclusive

        offset = parse_timedelta(window_info.offset)
        node.endpoint_expr = (st_inclusive, end_event, end_inclusive, offset)

        constraints = {}
        if window_info.excludes:
            for each_exclusion in window_info.excludes:
                if each_exclusion["predicate"]:
                    constraints[f"is_{each_exclusion['predicate']}"] = (None, 0)

        if window_info.includes:
            for each_inclusion in window_info.includes:
                if each_inclusion["predicate"]:
                    constraints[f"is_{each_inclusion['predicate']}"] = (
                        int(each_inclusion["min"])
                        if "min" in each_inclusion and each_inclusion["min"] is not None
                        else None,
                        int(each_inclusion["max"])
                        if "max" in each_inclusion and each_inclusion["max"] is not None
                        else None,
                    )
        node.constraints = constraints

        if window_info.start:
            node_root = next(
                (substring for substring in windows if substring in window_info.start),
                None,
            )
        elif window_info.end:
            node_root = next(
                (substring for substring in windows if substring in window_info.end),
                None,
            )

        if not node_root:
            nodes[window_name] = node
        elif node_root not in nodes:
            windows.append(window_name)
        else:
            node.parent = nodes[node_root]
            nodes[window_name] = node

    root = next(iter(nodes.values())).root

    # if event_bound window follows temporal window, the st_inclusive parameter is set to True
    for each_node in preorder_iter(root):
        if (
            each_node.parent
            and isinstance(each_node.parent.endpoint_expr[1], timedelta)
            and isinstance(each_node.endpoint_expr[1], str)
        ):
            each_node.endpoint_expr = (
                True,
                each_node.endpoint_expr[1],
                each_node.endpoint_expr[2],
                each_node.endpoint_expr[3],
            )
    return root
