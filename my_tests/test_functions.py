import pandas as pd
import pickle
import numpy as np
import sys
sys.path.append('../amlfbp')

def can_run():
    try :
        from __main__ import main
        main()
        return True
    except:
        return False

def detect_framework(path,fileName):
    from __main__ import autodetect_framework
    return (autodetect_framework(path) in fileName)

def can_configurate():
    try :
        from config import main
        main()
        return True
    except :
        return False