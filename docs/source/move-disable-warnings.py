import shutil
from pathlib import Path

p = Path("/home/docs/.ipython/profile_default/startup")
if p.exists():
    print(f"Adding disable-warnings.py and set-headers.py to {str(p)}")
    p.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        "disable-warnings.py",
        "/home/docs/.ipython/profile_default/startup/",
    )
    shutil.copy("set-headers.py", "/home/docs/.ipython/profile_default/startup")
