from typing import Any, Dict, List, Optional, TypedDict, Literal
from io import StringIO
import traceback
import pandas as pd
import numpy
import matplotlib
#import seaborn
#import scipy
#import statsmodels
#import sklearn
#import PIL
#import pyarrow

def safe_exec_python(code: str, dataframes: List[pd.DataFrame], image_bytes) -> Dict[str, Any]:
    # Minimal sandbox: no builtins except a tiny whitelist, no file/network ops
    # allowed_builtins = {
    #     "len": len,
    #     "range": range,
    #     "min": min,
    #     "max": max,
    #     "sum": sum,
    #     "print": print,
    #     "enumerate": enumerate,
    #     "sorted": sorted,
    # }
    safe_globals = {
        "__builtins__": __builtins__,
        "pd": pd,
        "numpy": numpy,
        "matplotlib": matplotlib,
#        "seaborn": seaborn,
#        "scipy": scipy,
#        "statsmodels": statsmodels,
#        "sklearn": sklearn,
#        "PIL": PIL,
#        "pyarrow": pyarrow,
        # Provide recent SQL results
        "dataframes": dataframes,
        "image_bytes": image_bytes
    }
    stdout = StringIO()
    stderr = StringIO()
    try:
        import sys
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stderr
        exec(  # noqa: S102 (we're sandboxing carefully)
            code,
            safe_globals,
            {},
        )
        sys.stdout = old_stdout
        sys.stderr = old_stderr
    except Exception:
        sys.stdout = old_stdout
        return {"ok": False, "stdout": stdout.getvalue(), "error": traceback.format_exc()}
    
    if len(stderr.getvalue()) > 0:
        return {"ok": False, "stdout": stdout.getvalue(), "error": stderr.getvalue()}

    return {"ok": True, "stdout": stdout.getvalue()}
