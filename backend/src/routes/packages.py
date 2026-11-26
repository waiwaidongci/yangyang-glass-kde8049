from typing import Annotated

from fastapi import APIRouter, Body

from src.controllers.dependencies import Actor
from src.services import packages

router = APIRouter(prefix='/packages')


@router.post('/merge')
def merge(actor: Actor, payload: Annotated[dict, Body()]):
    """在线合并各台模型包；断网复测台用 scripts/merge_packages.py 做同一件事。"""
    return packages.merge(payload)
