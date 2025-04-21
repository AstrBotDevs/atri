from astrbot.core.provider.func_tool_manager import FuncCall

EXTRACT_ENTITIES_TOOL = {
    "name": "extract_entities",
    "description": "Extract entities and types from user's query.",
    "parameters": {
        "type": "object",
        "properties": {
            "entities": {
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "The name of the entity.",
                        },
                        "type": {
                            "type": "string",
                            "description": "The type of the entity.",
                        },
                    },
                    "required": ["entity", "entity_type"],
                    "additionalProperties": False,
                },
                "description": "An array of entities with their types.",
                "type": "array",
            }
        },
        "required": ["entities"],
        "additionalProperties": False,
    },
}

BUILD_RELATIONS_TOOL = {
    "name": "build_relations",
    "description": "Build relations between entities.",
    "parameters": {
        "type": "object",
        "properties": {
            "relations": {
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "description": "The source entity of the relation.",
                        },
                        "target": {
                            "type": "string",
                            "description": "The target entity of the relation.",
                        },
                        "relation_type": {
                            "type": "string",
                            "description": "The type of the relation.",
                        },
                    },
                    "required": ["source_entity", "target_entity", "relation_type"],
                    "additionalProperties": False,
                },
                "description": "An array of relations between entities.",
                "type": "array",
            }
        },
        "required": ["relations"],
        "additionalProperties": False,
    },
}


def create_astrbot_func_mgr(tools: list):
    """创建一个函数管理器"""
    functool = FuncCall()
    for tool in tools:
        functool.add_func_from_raw_tool_def(tool)
    return functool
