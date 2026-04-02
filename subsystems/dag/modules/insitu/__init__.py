
# from .loader import InSituLoader # noqa: F821 
# from .aligner import TemporalAligner # noqa: F821 

# from subsystems.dag.modules.insitu.loader import InSituLoader
# from subsystems.dag.modules.insitu.aligner import TemporalAligner

# subsystems/dag/modules/insitu/__init__.py

print("DEBUG: loading insitu package")

try:
    from .loader import InSituLoader
    print("DEBUG: InSituLoader loaded")
except Exception as e:
    print("DEBUG ERROR loader:", e)

try:
    from .aligner import TemporalAligner
    print("DEBUG: TemporalAligner loaded")
except Exception as e:
    print("DEBUG ERROR aligner:", e)

__all__ = ["InSituLoader", "TemporalAligner"]
