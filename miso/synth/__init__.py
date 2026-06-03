"""Arm-A synthetic corpus generator (see eval_design_v1.md §3).

Clean, course-structured open-courseware → telegraphic 'student note' text →
realism-rendered handwritten page images, with the clean source emitted as exact
gold. The source IS the ground truth, so no annotation is needed for this arm.

    python -m miso.synth --course biology --limit 20
"""
