from models import OpenAiBaseModel
from config import settings
from utils import types as t

from pathlib import Path

class KimiK2p6(OpenAiBaseModel):
    def __init__(self,
                 model_name="kimi-k2.6",
                 system_prompt="Ты полезный ИИ ассистент",
                 base_url="https://api.moonshot.ai/v1",
                 api_key=settings.KIMI_API_KEY):
        super().__init__(model_name, system_prompt, base_url, api_key, True)

    def _process_asset(self, asset: t.Asset) -> None | dict:
        if asset.type == "image":
            if asset.size_bytes < 20 * 1024 * 1024: # В доках нет точного указания на размер файла, но допустим 20 Мб, как в genai
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{asset.mime_type};base64,{asset.data_base64}",
                    },
                }
            else:
                file_object = self.client.files.create(file=asset.local_path, purpose="image")
                return {
                    "type": "image_url",
                    "image_url": {
                        "url": f"ms://{file_object.id}"
                    }
                }
        elif asset.type == "video":
            if asset.size_bytes < 20 * 1024 * 1024:
                return {
                    "type": "video_url",
                    "video_url": {
                        "url": f"data:{asset.mime_type};base64,{asset.data_base64}",
                    },
                }
            else:
                file_object = self.client.files.create(file=asset.local_path, purpose="video")
                return {
                    "type": "video_url",
                    "video_url": {
                        "url": f"ms://{file_object.id}"
                    }
                }
        elif asset.type == "document":
            pass
            # В kimi api документ принимается от имени системы. То есть обработкой должен заниматься _convert_history_from_umf
            # if not asset.ocr_text:
            #     file_object = self.client.files.create(file=asset.local_path, purpose="file-extract")
            #     file_text = self.client.files.content(file_id=file_object.id).text
            #     asset.ocr_text = file_text # Вот эта строка должна влиять на сам ассет, как если бы была передана ссылка на объект, а не копия