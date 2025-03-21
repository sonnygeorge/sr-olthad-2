from agent_framework.lms import OpenAIInstructLm
from agent_framework.schema import InstructLm


class SrOlthadCfg:
    JSON_DUMPS_INDENT = 3


class AttemptSummarizerCfg:
    MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 1  # 3
    INSTRUCT_LM: InstructLm = OpenAIInstructLm(model="gpt-3.5-turbo")
    PROMPTS_VERSION = "1.0"


class BacktrackerCfg:
    N_CALLS_FOR_VOTING: int = 3  # TODO: implement?
    MAX_ASYNC_CALL_FOR_VOTING: int = 5

    class ExhaustiveEffortClf:
        N_CALLS_FOR_VOTING: int = 1
        MAX_ASYNC_CALL_FOR_VOTING: int = 5
        MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 1  # 3
        INSTRUCT_LM: InstructLm = OpenAIInstructLm(
            model="gpt-3.5-turbo"  # "gpt-4o-mini-2024-07-18"
        )
        PROMPTS_VERSION = "1.0"

    class MostWorthwhilePursuitClfCfg:
        N_CALLS_FOR_VOTING: int = 1
        MAX_ASYNC_CALL_FOR_VOTING: int = 5
        MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 1  # 3
        INSTRUCT_LM: InstructLm = OpenAIInstructLm(model="gpt-4o-mini-2024-07-18")
        PROMPTS_VERSION = "1.0"

    class PartialSuccessClfCfg:
        N_CALLS_FOR_VOTING: int = 1
        MAX_ASYNC_CALL_FOR_VOTING: int = 5
        MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 1  # 3
        INSTRUCT_LM: InstructLm = OpenAIInstructLm(model="gpt-3.5-turbo")
        PROMPTS_VERSION = "1.0"

    class SuccessfulCompletionClfCfg:
        N_CALLS_FOR_VOTING: int = 1
        MAX_ASYNC_CALL_FOR_VOTING: int = 5
        MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 1  # 3
        INSTRUCT_LM: InstructLm = OpenAIInstructLm(
            model="gpt-3.5-turbo"  # "gpt-4o-mini-2024-07-18"
        )
        PROMPTS_VERSION = "1.0"


class ForgetterCfg:
    MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 1  # 3
    INSTRUCT_LM: InstructLm = OpenAIInstructLm(model="gpt-3.5-turbo")
    PROMPTS_VERSION = "1.0"


class PlannerCfg:
    MAX_TRIES_TO_GET_VALID_LM_RESPONSE: int = 3
    INSTRUCT_LM: InstructLm = OpenAIInstructLm(model="gpt-3.5-turbo")
    PROMPTS_VERSION = "1.0"
