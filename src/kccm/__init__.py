"""KCCM：关键词缓存上下文管理。"""

from .cache import KeywordCache
from .composer import ComposeResult, Composer, ContextBudgetExceeded, estimate_tokens
from .config import Config, load_config
from .extractor import Extractor
from .llm import LLMClient, OpenAICompatibleLLM, RuleLLM, get_llm
from .models import CacheEntry, Candidate, HitKind, SlotRole, UpsertKind
from .session import Session, TurnResult
from .tagger import LexiconTagger, get_tagger
from .trace import Tracer

__version__ = "2.0.0"

__all__ = [
    "CacheEntry",
    "Candidate",
    "ComposeResult",
    "Composer",
    "Config",
    "ContextBudgetExceeded",
    "Extractor",
    "HitKind",
    "KeywordCache",
    "LLMClient",
    "LexiconTagger",
    "OpenAICompatibleLLM",
    "RuleLLM",
    "Session",
    "SlotRole",
    "Tracer",
    "TurnResult",
    "UpsertKind",
    "estimate_tokens",
    "get_llm",
    "get_tagger",
    "load_config",
]
