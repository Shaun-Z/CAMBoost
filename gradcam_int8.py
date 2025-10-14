#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin CLI wrapper for the CAMBoost INT8 GradCAM demos.

The heavy lifting lives in the reusable `camboost.int8_gradcam` package.
"""

from camboost.int8_gradcam.demo import main


if __name__ == "__main__":
    main()
