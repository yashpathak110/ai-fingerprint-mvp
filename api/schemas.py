from pydantic import BaseModel;
class PredictRequest(BaseModel): text: str; 
prompt_type: str = "general"
