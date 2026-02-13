"""Image Captioner transform for enriching chunks with image descriptions."""

from pathlib import Path
from typing import List, Optional

from src.core.settings import Settings
from src.core.types import Chunk
from src.core.trace.trace_context import TraceContext
from src.ingestion.transform.base_transform import BaseTransform
from src.libs.llm.base_vision_llm import BaseVisionLLM, ImageInput
from src.libs.llm.llm_factory import LLMFactory
from src.observability.logger import get_logger

logger = get_logger(__name__)

class ImageCaptioner(BaseTransform):
    """Generates captions for images referenced in chunks using Vision LLM.
    
    This transform identifies chunks containing image references, uses a Vision LLM
    to generate descriptive captions, and enriches the chunk text/metadata with
    these captions to improve retrieval for visual content.
    """
    
    def __init__(
        self, 
        settings: Settings, 
        llm: Optional[BaseVisionLLM] = None
    ):
        self.settings = settings
        self.llm = None
        
        # Check if vision LLM is enabled in settings
        if self.settings.vision_llm and self.settings.vision_llm.enabled:
             try:
                 self.llm = llm or LLMFactory.create_vision_llm(settings)
             except Exception as e:
                 logger.error(f"Failed to initialize Vision LLM: {e}")
                 # We don't raise here to allow pipeline to continue without captioning
                 # effectively falling back to no-op for this transform
        else:
             logger.warning("Vision LLM is disabled or not configured. ImageCaptioner will skip processing.")
        
        self.prompt = self._load_prompt()
        
    def _load_prompt(self) -> str:
        """Load the image captioning prompt from configuration."""
        # Assuming standard relative path. In production, logic might be robust.
        prompt_path = Path("config/prompts/image_captioning.txt")
        if prompt_path.exists():
            return prompt_path.read_text(encoding="utf-8").strip()
        return "Describe this image in detail for indexing purposes."

    def transform(
        self,
        chunks: List[Chunk],
        trace: Optional[TraceContext] = None
    ) -> List[Chunk]:
        """Process chunks and add captions for referenced images."""
        if not self.llm:
            return chunks
            
        processed_chunks = []
        for chunk in chunks:
            # Check for images in metadata (populated by Loader)
            if not chunk.metadata or "images" not in chunk.metadata:
                processed_chunks.append(chunk)
                continue
                
            images = chunk.metadata.get("images", [])
            if not images:
                processed_chunks.append(chunk)
                continue
                
            # Process each image in the chunk
            new_text = chunk.text
            captions = []
            
            # Using trace context if provided
            t = args = kwargs = None
            if trace:
                 # Manually handling trace logic simulation as TraceContext usage differs
                 pass
            
            for img_meta in images:
                img_path = img_meta.get("path")
                img_id = img_meta.get("id")
                
                if not img_path or not Path(img_path).exists():
                        logger.warning(f"Image path not found: {img_path}")
                        continue
                
                try:
                    image_input = ImageInput(path=img_path)
                    
                    # Pass trace if supported by chat_with_image
                    response = self.llm.chat_with_image(
                        text=self.prompt,
                        image=image_input,
                        trace=trace
                    )
                    caption = response.content
                    captions.append({"id": img_id, "caption": caption})
                    
                    # Replace placeholder or append
                    # Standard placeholder: [IMAGE: {id}]
                    placeholder = f"[IMAGE: {img_id}]"
                    replacement = f"[IMAGE: {img_id}]\n(Description: {caption})"
                    
                    if placeholder in new_text:
                        new_text = new_text.replace(placeholder, replacement)
                    else:
                        # Append if placeholder missing but metadata exists
                        new_text += f"\n\n(Image Description for {img_id}: {caption})"
                        
                except Exception as e:
                    logger.error(f"Failed to caption image {img_path}: {e}")
                    # Continue to next image
                    
            chunk.text = new_text
            # Store captions in metadata
            if "image_captions" not in chunk.metadata:
                chunk.metadata["image_captions"] = []
            chunk.metadata["image_captions"].extend(captions)
            
            processed_chunks.append(chunk)
            
        return processed_chunks
