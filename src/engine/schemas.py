from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator


class SWOTBase(BaseModel):
    strengths: List[str] = Field(..., min_length=1)
    weaknesses: List[str] = Field(..., min_length=1)
    opportunities: List[str] = Field(..., min_length=1)
    threats: List[str] = Field(..., min_length=1)

    # If enabled, we store per-quadrant assumptions (preferred) or a global list
    assumptions: Optional[Union[Dict[str, List[str]], List[str]]] = None

    @field_validator("strengths", "weaknesses", "opportunities", "threats")
    @classmethod
    def no_empty_strings(cls, v: List[str]) -> List[str]:
        cleaned = [s.strip() for s in v if isinstance(s, str) and s.strip()]
        if not cleaned:
            raise ValueError("Quadrant must contain non-empty bullet strings.")
        return cleaned


class SWOTResult(SWOTBase):
    raw_model_output: str = ""
    validation_warnings: List[str] = []
    key_insights: List[str] = []
    action_suggestions: List[str] = []
