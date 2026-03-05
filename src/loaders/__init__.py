# Import loaders so they register themselves via LoaderFactory.register(...)
from .qa_loader import QALoader  # noqa: F401
from .mcq_loader import MCQLoader  # noqa: F401
from .math500_loader import Math500Loader  # noqa: F401
from .code_loader import CodeUnitTestLoader  # noqa: F401
