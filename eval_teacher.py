"""Evaluate the frozen checkpoint's teacher branch using original + FlashSR audio."""

from utils import evaluation_main

if __name__ == "__main__":
    evaluation_main("teacher")
