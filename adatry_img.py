import io
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict
import logging
import httpx
import modal
import json

# ============================================================================
# IMAGE CONFIGURATION
# ============================================================================

flux_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .apt_install("git")
    .pip_install("uv")
    .run_commands(
        "uv pip install --system --compile-bytecode --index-strategy unsafe-best-match "
        "accelerate~=1.8.1 "
        "git+https://github.com/huggingface/diffusers.git@00f95b9755718aabb65456e791b8408526ae6e76 "
        "huggingface-hub[hf-transfer]~=0.33.1 "
        "Pillow~=11.2.1 "
        "safetensors~=0.5.3 "
        "transformers~=4.53.0 "
        "sentencepiece~=0.2.0 "
        "fastapi[standard]==0.115.4 "
        "python-multipart==0.0.12 "
        "cloudinary "
        "httpx "
        "--extra-index-url https://download.pytorch.org/whl/cu128"
    )
)

# ============================================================================
# VOLUMES & SECRETS
# ============================================================================

FLUX_CACHE_DIR = Path("/cache")
flux_cache_volume = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
flux_volumes = {FLUX_CACHE_DIR: flux_cache_volume}

secrets = [modal.Secret.from_name("custom-secret", required_keys=[
    "HF_TOKEN", 
    "BEARER_TOKEN",
    "CLOUDINARY_CLOUD_NAME",
    "CLOUDINARY_API_KEY",
    "CLOUDINARY_API_SECRET"
])]

# ============================================================================
# CONSTANTS
# ============================================================================

FLUX_MODEL_NAME = "black-forest-labs/FLUX.1-Kontext-dev"
FLUX_MODEL_REVISION = "f9fdd1a95e0dfd7653cb0966cda2486745122695"

# ============================================================================
# PAKAIAN ADAT DATABASE
# ============================================================================

PAKAIAN_ADAT_PROMPTS = {
    "ulee_balang": {
        "name": "Ulee Balang (Aceh)",
        "provinsi": "Aceh",
        "prompt_template": """Apply the traditional clothing 'Ulee Balang' from Aceh to this image. Black outfit with luxury gold embroidery. Features items like Meukeutop hat, rencong, heavy jewelry, or majestic headdress depending on subject. Keep original face and pose."""
    },
    "bundo_kanduang": {
        "name": "Bundo Kanduang (Sumatera Barat)",
        "provinsi": "Sumatera Barat",
        "prompt_template": """Apply the traditional clothing 'Bundo Kanduang' from West Sumatra to this image. Red gold songket fabric. Characteristic items include Deta hat, or the buffalo horn headdress (Tengkuluk) and gold necklace. Keep original face and pose."""
    },
    "ulos": {
        "name": "Ulos (Sumatera Utara)",
        "provinsi": "Sumatera Utara",
        "prompt_template": """Apply the traditional clothing 'Ulos' from North Sumatra to this image. Red gold woven Ulos sash. Attire may include black suit and Batak hat, or long kebaya and red Sortali headband. Keep original face and pose."""
    },
    "aesan_gede": {
        "name": "Aesan Gede (Sumatera Selatan)",
        "provinsi": "Sumatera Selatan",
        "prompt_template": """Apply the traditional clothing 'Aesan Gede' from South Sumatra to this image. Maroon gold songket style. Features majestic gold crowns (Kopiah Mas or large Karsuhun) and gold chest cover ornaments. Keep original face and pose."""
    },
    "teluk_belanga": {
        "name": "Teluk Belanga (Kepulauan Riau)",
        "provinsi": "Kepulauan Riau",
        "prompt_template": """Apply the traditional clothing 'Teluk Belanga' from Riau Islands to this image. Bright solid colors Malay dress. Attire includes Teluk Belanga with peci hat and songket waist cloth, or Kebaya Laboh with head shawl. Keep original face and pose."""
    },
    "kebaya_laboh_kurung": {
        "name": "Kebaya Laboh dan Kurung Cekak Musang (Riau)",
        "provinsi": "Riau",
        "prompt_template": """Apply the traditional clothing from Riau to this image. Satin fabric. Features Cekak Musang suit with Tanjak headdress, or long Kebaya Laboh with woven cloth. Keep original face and pose."""
    },
    "baju_betabur": {
        "name": "Baju Betabur (Bengkulu)",
        "provinsi": "Bengkulu",
        "prompt_template": """Apply the traditional clothing 'Baju Betabur' from Bengkulu to this image. Velvet with gold metal plates. Includes headwear like pointed Detar hat or Singal crown with gold chest ornament. Keep original face and pose."""
    },
    "baju_kurung_jambi": {
        "name": "Baju Kurung (Jambi)",
        "provinsi": "Jambi",
        "prompt_template": """Apply the traditional clothing from Jambi to this image. Velvet gold embroidery. Attire includes Lacak hat and songket sarong, or Baju Kurung Tanggung with Pesangkon headdress. Keep original face and pose."""
    },
    "paksian": {
        "name": "Paksian (Bangka Belitung)",
        "provinsi": "Bangka Belitung",
        "prompt_template": """Apply the traditional clothing 'Paksian' from Bangka Belitung to this image. Red purple Malay Chinese style. Features include Sungkon turban or gold Paksian crown with baju kurung. Keep original face and pose."""
    },
    "tulang_bawang": {
        "name": "Tulang Bawang (Lampung)",
        "provinsi": "Lampung",
        "prompt_template": """Apply the traditional clothing 'Tulang Bawang' from Lampung to this image. Gold Tapis fabric. Key items include closed suit with gold hat, or large gold Siger crown with white velvet dress. Keep original face and pose."""
    },
    "pangsi": {
        "name": "Pangsi (Banten)",
        "provinsi": "Banten",
        "prompt_template": """Apply the traditional clothing 'Pangsi' from Banten to this image. Warrior style attire. Includes black Pangsi suit with Lomar headband, or simple kebaya with batik cloth. Keep original face and pose."""
    },
    "kebaya_encim": {
        "name": "Kebaya Encim (DKI Jakarta)",
        "provinsi": "DKI Jakarta",
        "prompt_template": """Apply the traditional clothing 'Betawi' from Jakarta to this image. Attire may be white Sadariah shirt with scarf and peci, or bright Kebaya Encim with floral embroidery and batik skirt. Keep original face and pose."""
    },
    "kebaya_sunda": {
        "name": "Kebaya Sunda (Jawa Barat)",
        "provinsi": "Jawa Barat",
        "prompt_template": """Apply the traditional clothing from West Java to this image. Elegant style with batik. Features Beskap suit with Blangkon, or modern brocade Kebaya with batik skirt. Keep original face and pose."""
    },
    "kesatrian_ageng": {
        "name": "Kesatrian Ageng (DI Yogyakarta)",
        "provinsi": "Daerah Istimewa Yogyakarta",
        "prompt_template": """Apply the traditional clothing 'Kesatrian Ageng' from Yogyakarta to this image. Royal palace attire with batik. Includes Surjan jacket with Blangkon, or black velvet Kebaya with gold embroidery and hair bun. Keep original face and pose."""
    },
    "jawi_jangkep": {
        "name": "Jawi Jangkep (Jawa Tengah)",
        "provinsi": "Jawa Tengah",
        "prompt_template": """Apply the traditional clothing 'Jawi Jangkep' from Central Java to this image. Solo style attire with batik. Features black Beskap with Blangkon and keris, or long velvet Kebaya with hair bun. Keep original face and pose."""
    },
    "pesaan": {
        "name": "Pesa’an (Jawa Timur)",
        "provinsi": "Jawa Timur",
        "prompt_template": """Apply the traditional clothing 'Pesa-an' from East Java to this image. Madura style. Includes striped shirt with loose black suit and Udeng, or red Kebaya with batik sarong. Keep original face and pose."""
    },
    "payas_agung": {
        "name": "Payas Agung (Bali)",
        "provinsi": "Bali",
        "prompt_template": """Apply the traditional clothing 'Payas Agung' from Bali to this image. Gold prada fabric luxury attire. Features Songket Udeng and keris, or high gold flower headdress (Gelung Agung). Keep original face and pose."""
    },
    "pegon": {
        "name": "Pegon (Nusa Tenggara Barat)",
        "provinsi": "Nusa Tenggara Barat",
        "prompt_template": """Apply the traditional clothing from West Nusa Tenggara to this image. Sasak attire with songket. Includes Pegon suit with Sapuk headband, or black Lambung shirt with songket sash. Keep original face and pose."""
    },
    "amarasi": {
        "name": "Amarasi (Nusa Tenggara Timur)",
        "provinsi": "Nusa Tenggara Timur",
        "prompt_template": """Apply the traditional clothing 'Amarasi' from East Nusa Tenggara to this image. Timor woven ikat fabric with geometric patterns. Features woven sash, or woven chest wrap with crescent metal headdress. Keep original face and pose."""
    },
    "king_bibinge_baba": {
        "name": "King Bibinge dan King Baba (Kalimantan Barat)",
        "provinsi": "Kalimantan Barat",
        "prompt_template": """Apply the traditional clothing 'Dayak' from West Kalimantan to this image. Tree bark fabric, beads. Tall hornbill feather headdress, tribal painted vest, arm accessories. Keep original face and pose."""
    },
    "upak_nyamu": {
        "name": "Upak Nyamu (Kalimantan Tengah)",
        "provinsi": "Kalimantan Tengah",
        "prompt_template": """Apply the traditional clothing 'Upak Nyamu' from Central Kalimantan to this image. Natural tree bark vest, bead necklace, tribal bracelets. Natural look. Keep original face and pose."""
    },
    "taa_sapei_sapaq": {
        "name": "Ta’a dan Sapei Sapaq (Kalimantan Utara)",
        "provinsi": "Kalimantan Utara",
        "prompt_template": """Apply the traditional clothing 'Dayak Kenyah' from North Kalimantan to this image. Black velvet, colorful beads. Includes shield and mandau, or fully beaded vest (Ta'a/Sapei Sapaq). Keep original face and pose."""
    },
    "bagajah_gamuling": {
        "name": "Bagajah Gamuling (Kalimantan Selatan)",
        "provinsi": "Kalimantan Selatan",
        "prompt_template": """Apply the traditional clothing 'Bagajah Gamuling' from South Kalimantan to this image. Banjar wedding attire covered in fresh jasmine flower garlands and glittering cloth. Includes shaking flower crown. Keep original face and pose."""
    },
    "kustin": {
        "name": "Kustin (Kalimantan Timur)",
        "provinsi": "Kalimantan Timur",
        "prompt_template": """Apply the traditional clothing 'Kustin' from East Kalimantan to this image. Black velvet, gold pasmen. Features tall round hat (setorong), or hair bun with metal flower ornaments. Keep original face and pose."""
    },
    "pattuqduq_towaine": {
        "name": "Pattuqduq Towaine (Sulawesi Barat)",
        "provinsi": "Sulawesi Barat",
        "prompt_template": """Apply the traditional clothing 'Mandar' from West Sulawesi to this image. Mandar style silk sarong. Attire includes Closed suit, or Baju Pokko with gold accessories. Keep original face and pose."""
    },
    "nggembe": {
        "name": "Nggembe (Sulawesi Tengah)",
        "provinsi": "Sulawesi Tengah",
        "prompt_template": """Apply the traditional clothing 'Nggembe' from Central Sulawesi to this image. Donggala cloth. Features Koje shirt, or Nggembe square collar shirt with gold chest plate. Keep original face and pose."""
    },
    "laku_tepu": {
        "name": "Laku Tepu (Sulawesi Utara)",
        "provinsi": "Sulawesi Utara",
        "prompt_template": """Apply the traditional clothing 'Laku Tepu' from North Sulawesi to this image. Long fiber robe with bright colors and geometric patterns. Includes pointed Paporong hat. Keep original face and pose."""
    },
    "babu_nggawi": {
        "name": "Babu Nggawi (Sulawesi Tenggara)",
        "provinsi": "Sulawesi Tenggara",
        "prompt_template": """Apply the traditional clothing 'Babu Nggawi' from Southeast Sulawesi to this image. Bright colors, gold beads. Features pabele hat, or luxury beaded chest ornament. Keep original face and pose."""
    },
    "baju_bodo": {
        "name": "Baju Bodo (Sulawesi Selatan)",
        "provinsi": "Sulawesi Selatan",
        "prompt_template": """Apply the traditional clothing 'Baju Bodo' from South Sulawesi to this image. Bugis Makassar attire. Includes gold woven peci hat (songkok recca), or stiff transparent bright blouse (Baju Bodo) with silk sarong. Keep original face and pose."""
    },
    "biliu_makuta": {
        "name": "Biliu dan Makuta (Gorontalo)",
        "provinsi": "Gorontalo",
        "prompt_template": """Apply the traditional clothing from Gorontalo to this image. Purple dark red colors. Features tall Paluwala headdress, or shirt with long hanging gold beads (Biliu). Keep original face and pose."""
    },
    "cele": {
        "name": "Cele (Maluku)",
        "provinsi": "Maluku",
        "prompt_template": """Apply the traditional clothing 'Baju Cele' from Maluku to this image. Red white plaid Baju Cele style. Attire includes jacket and sarong, or white kebaya and woven sarong. Keep original face and pose."""
    },
    "manteren_lamo": {
        "name": "Manteren Lamo (Maluku Utara)",
        "provinsi": "Maluku Utara",
        "prompt_template": """Apply the traditional clothing 'Manteren Lamo' from North Maluku to this image. Red royal robe with large gold buttons. Majestic style, includes sultanate headdress. Keep original face and pose."""
    },
    "ewer": {
        "name": "Ewer (Papua Barat)",
        "provinsi": "Papua Barat",
        "prompt_template": """Apply the traditional clothing 'Ewer' from West Papua to this image. Straw tree bark material. Large bead necklace, cassowary feather headdress. Tribal natural look. Keep original face and pose."""
    },
    "koteka_holim": {
        "name": "Koteka (Papua)",
        "provinsi": "Papua",
        "prompt_template": """Apply the traditional clothing from Papua to this image. Papua tribal attire. Features Bird of Paradise crown, nose tusk, painted body, or tassel skirt with Noken bag on head. Keep original face and pose."""
    },
    "pummi": {
        "name": "Pummi (Papua Selatan)",
        "provinsi": "Papua Selatan",
        "prompt_template": """Apply the traditional clothing 'Pummi' from South Papua to this image. Sago leaf tassel skirt. Intricate bamboo white feather headdress, animal tooth necklace. Ethnic detail. Keep original face and pose."""
    },
    "yokal": {
        "name": "Yokal (Papua Tengah)",
        "provinsi": "Papua Tengah",
        "prompt_template": """Apply the traditional clothing 'Yokal' from Central Papua to this image. Features reddish tree bark skirt (Yokal), or tribal accessories, noken bag, and feather headdress. Keep original face and pose."""
    },
    "holim_dani": {
        "name": "Holim (Papua Pegunungan)",
        "provinsi": "Papua Pegunungan",
        "prompt_template": """Apply the traditional clothing 'Dani Tribe' from Highland Papua to this image. Natural material attire. Includes body paint and feather headdress, or straw skirt. Spear prop. Keep original face and pose."""
    },
    "ewer_tehit": {
        "name": "Ewer Tehit (Papua Barat Daya)",
        "provinsi": "Papua Barat Daya",
        "prompt_template": """Apply the traditional clothing 'Ewer Tehit' from Southwest Papua to this image. Long dried sago leaf skirt. Bound cloth top. Woven feather headdress. Keep original face and pose."""
    }
}

# ============================================================================
# LOGGING SETUP
# ============================================================================

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# MODAL APP
# ============================================================================

app = modal.App("pakaian-adat-generator")

with flux_image.imports():
    import torch
    import os
    from diffusers import FluxKontextPipeline
    from diffusers.utils import load_image
    from PIL import Image
    from io import BytesIO
    import cloudinary
    import cloudinary.uploader
    import httpx
    
# ============================================================================
# FLUX KONTEXT MODEL CLASS
# ============================================================================

@app.cls(
    image=flux_image,
    cpu="0.5",
    memory="2GiB",
    gpu="L40s",
    volumes=flux_volumes,
    secrets=secrets,
    scaledown_window=120,
    timeout=900,
)
class FluxKontextModel:
    """
    Flux Kontext model optimized untuk pakaian adat generation.
    
    Features:
    - High-quality image-to-image editing
    - Context-aware costume changes
    - Face and pose preservation
    - Cloudinary upload integration
    """
    
    @modal.enter()
    def enter(self):
        """Initialize model and Cloudinary on container startup."""
        logger.info(f"[FLUX] Loading {FLUX_MODEL_NAME}...")
        
        dtype = torch.bfloat16
        self.device = "cuda"
        
        # Load Flux Kontext Pipeline
        self.kontext_pipe = FluxKontextPipeline.from_pretrained(
            FLUX_MODEL_NAME,
            revision=FLUX_MODEL_REVISION,
            torch_dtype=dtype,
            cache_dir=FLUX_CACHE_DIR,
            token=os.environ.get("HF_TOKEN"),
        ).to(self.device)
        
        logger.info("[FLUX] Model loaded successfully")
        
        # Configure Cloudinary
        cloudinary.config( 
            cloud_name=os.environ["CLOUDINARY_CLOUD_NAME"], 
            api_key=os.environ["CLOUDINARY_API_KEY"], 
            api_secret=os.environ["CLOUDINARY_API_SECRET"],
            secure=True
        )
        logger.info("[CLOUDINARY] SDK configured")

    @modal.method()
    def generate_pakaian_adat(
        self,
        image_bytes: bytes,
        pakaian_id: str,
        guidance_scale: float = 4.0,
        num_inference_steps: int = 30,
        seed: Optional[int] = None,
    ) -> Dict:
        """
        Generate pakaian adat image with optimized settings.
        
        Args:
            image_bytes: Input portrait image
            pakaian_id: ID pakaian adat (e.g., "ulee_balang")
            guidance_scale: CFG scale (4.0 optimal untuk detail preservation)
            num_inference_steps: Steps (30 untuk quality balance)
            seed: Random seed for reproducibility
            
        Returns:
            Dict containing:
                - url: Cloudinary URL
                - pakaian_name: Nama pakaian adat
                - provinsi: Asal provinsi
                - prompt_used: Prompt yang digunakan
        """
        # Validate pakaian_id
        if pakaian_id not in PAKAIAN_ADAT_PROMPTS:
            raise ValueError(f"Invalid pakaian_id: {pakaian_id}")
        
        pakaian_data = PAKAIAN_ADAT_PROMPTS[pakaian_id]
        prompt = pakaian_data["prompt_template"]
        
        logger.info(f"[FLUX] Generating {pakaian_data['name']}...")
        logger.info(f"[FLUX] Steps: {num_inference_steps}, CFG: {guidance_scale}")
        
        # Setup generator
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.device)
        
        # Load input image
        init_image = load_image(Image.open(BytesIO(image_bytes)))
        orig_width, orig_height = init_image.size
        
        # Calculate optimal dimensions (maintain aspect ratio, divisible by 8)
        max_dimension = 1024
        if orig_width > orig_height:
            width = max_dimension
            height = int((orig_height / orig_width) * max_dimension)
        else:
            height = max_dimension
            width = int((orig_width / orig_height) * max_dimension)
        
        # Ensure divisible by 8
        width = (width // 8) * 8
        height = (height // 8) * 8
        
        logger.info(f"[FLUX] Output dimensions: {width}x{height}")
        
        # Run inference with optimized settings
        image = self.kontext_pipe(
            width=width,
            height=height,
            image=init_image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            generator=generator,
        ).images[0]
        
        # Convert to bytes
        byte_stream = BytesIO()
        image.save(byte_stream, format="PNG", quality=95)
        
        logger.info("[FLUX] Inference completed. Uploading to Cloudinary...")
        
        try:
            byte_stream.seek(0)
            upload_result = cloudinary.uploader.upload(
                byte_stream,
                folder=f"pakaian_adat/{pakaian_data['provinsi'].lower().replace(' ', '_')}",
                resource_type="image",
                public_id=f"{pakaian_id}_{seed if seed else 'random'}",
                overwrite=False
            )
            
            secure_url = upload_result.get('secure_url')
            if not secure_url:
                raise Exception("Cloudinary upload failed: No secure_url returned")
                
            logger.info(f"[CLOUDINARY] Upload successful: {secure_url}")
            
            return {
                "url": secure_url,
                "pakaian_name": pakaian_data["name"],
                "provinsi": pakaian_data["provinsi"],
                "prompt_used": prompt,
                "dimensions": f"{width}x{height}",
                "seed": seed,
                "status": "success"
            }
            
        except Exception as e:
            logger.error(f"[CLOUDINARY] Upload failed: {e}")
            raise e

# ============================================================================
# FASTAPI WEB APPLICATION
# ============================================================================

@app.function(
    image=flux_image, 
    volumes=flux_volumes, 
    secrets=secrets, 
    cpu="0.5", 
    memory="2GiB", 
    timeout=900
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI application for Pakaian Adat Generator."""
    from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends
    from fastapi.responses import JSONResponse
    from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
    from fastapi.middleware.cors import CORSMiddleware
    from PIL import Image
    from io import BytesIO
    
    web_app = FastAPI(
        title="Pakaian Adat Generator API",
        description="Generate Indonesian traditional costume images using AI",
        version="1.0.0",
    )
    
    # CORS
    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # ========================================================================
    # ENDPOINT 1: Get Available Pakaian Adat List
    # ========================================================================
    
    @web_app.get("/pakaian-adat/list")
    async def get_pakaian_list():
        """
        Get list of all available pakaian adat.
        
        Returns list dengan grouping by provinsi.
        """
        pakaian_by_provinsi = {}
        
        for pakaian_id, data in PAKAIAN_ADAT_PROMPTS.items():
            provinsi = data["provinsi"]
            if provinsi not in pakaian_by_provinsi:
                pakaian_by_provinsi[provinsi] = []
            
            pakaian_by_provinsi[provinsi].append({
                "id": pakaian_id,
                "name": data["name"],
                "provinsi": provinsi
            })
        
        return JSONResponse(
            content={
                "total": len(PAKAIAN_ADAT_PROMPTS),
                "by_provinsi": pakaian_by_provinsi,
                "all_ids": list(PAKAIAN_ADAT_PROMPTS.keys())
            },
            status_code=200
        )
    
    # ========================================================================
    # ENDPOINT 2: Get Pakaian Detail
    # ========================================================================
    
    @web_app.get("/pakaian-adat/{pakaian_id}")
    async def get_pakaian_detail(pakaian_id: str):
        """Get detail of specific pakaian adat."""
        if pakaian_id not in PAKAIAN_ADAT_PROMPTS:
            raise HTTPException(
                status_code=404,
                detail=f"Pakaian '{pakaian_id}' not found"
            )
        
        data = PAKAIAN_ADAT_PROMPTS[pakaian_id]
        return JSONResponse(
            content={
                "id": pakaian_id,
                "name": data["name"],
                "provinsi": data["provinsi"],
                "prompt": data["prompt_template"]
            },
            status_code=200
        )
    
    # ========================================================================
    # ENDPOINT 3: Generate Pakaian Adat Image (MAIN)
    # ========================================================================
    
    @web_app.post("/generate")
    async def generate_pakaian_adat(
        image: UploadFile = File(..., description="Input portrait image (JPEG/PNG)"),
        pakaian_id: str = Form(..., description="ID pakaian adat (e.g., 'ulee_balang')"),
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        guidance_scale: float = Form(4.0, description="CFG scale (2-7, default 4)"),
        num_inference_steps: int = Form(30, description="Steps (20-50, default 30)"),
        seed: Optional[int] = Form(None, description="Random seed (optional)"),
    ):
        """
        Generate pakaian adat image.
        
        Main endpoint untuk user flow:
        1. Upload gambar portrait
        2. Pilih jenis pakaian (via pakaian_id)
        3. AI generate dengan prompt dari database
        4. Return Cloudinary URL
        
        Optimized settings:
        - CFG 4.0: Balance detail & coherence
        - Steps 30: Quality/speed tradeoff
        - Auto aspect ratio preservation
        """
        logger.info(f"[API] /generate called: pakaian_id={pakaian_id}")
        
        # Auth check
        if os.environ.get("BEARER_TOKEN", False):
            if not token or token.credentials != os.environ["BEARER_TOKEN"]:
                raise HTTPException(status_code=401, detail="Incorrect bearer token")
        
        # Validate pakaian_id
        if pakaian_id not in PAKAIAN_ADAT_PROMPTS:
            available_ids = list(PAKAIAN_ADAT_PROMPTS.keys())
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "Invalid pakaian_id",
                    "provided": pakaian_id,
                    "available": available_ids[:10],  # Show first 10
                    "total_available": len(available_ids),
                    "hint": "Use /pakaian-adat/list to see all options"
                }
            )
        
        try:
            # Read image
            image_bytes = await image.read()
            
            # Validate image
            if len(image_bytes) > 10 * 1024 * 1024:
                raise HTTPException(
                    status_code=400, 
                    detail="Image too large (max 10MB)"
                )
            
            pil_image = Image.open(BytesIO(image_bytes))
            width, height = pil_image.size
            
            logger.info(f"[API] Input image: {width}x{height}")
            
            # Validate aspect ratio (should be portrait-ish for best results)
            aspect_ratio = width / height
            if aspect_ratio < 0.4 or aspect_ratio > 2.5:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid aspect ratio ({width}:{height}). Use portrait or square images."
                )
            
            # Initialize model and generate
            model = FluxKontextModel()
            result = model.generate_pakaian_adat.remote(
                image_bytes=image_bytes,
                pakaian_id=pakaian_id,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
            )
            
            logger.info(f"[API] Generation completed: {result['url']}")
            
            return JSONResponse(
                content=result,
                status_code=200
            )
        
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"[API] Error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    # ========================================================================
    # ENDPOINT 4: Batch Generate (Multiple Pakaian)
    # ========================================================================
    
    @web_app.post("/generate-batch")
    async def generate_batch(
        image: UploadFile = File(...),
        pakaian_ids: str = Form(..., description="Comma-separated pakaian IDs"),
        token: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        guidance_scale: float = Form(4.0),
        num_inference_steps: int = Form(30),
    ):
        """
        Generate multiple pakaian adat from same input image.
        
        Use case: User wants to try multiple costumes on same photo.
        """
        logger.info(f"[API] /generate-batch called")
        
        # Auth
        if os.environ.get("BEARER_TOKEN", False):
            if not token or token.credentials != os.environ["BEARER_TOKEN"]:
                raise HTTPException(status_code=401, detail="Incorrect bearer token")
        
        # Parse pakaian_ids
        ids_list = [pid.strip() for pid in pakaian_ids.split(",")]
        
        if len(ids_list) > 5:
            raise HTTPException(
                status_code=400,
                detail="Maximum 5 pakaian in batch mode"
            )
        
        # Validate all IDs
        invalid_ids = [pid for pid in ids_list if pid not in PAKAIAN_ADAT_PROMPTS]
        if invalid_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid pakaian IDs: {invalid_ids}"
            )
        
        try:
            image_bytes = await image.read()
            
            # Validate image
            pil_image = Image.open(BytesIO(image_bytes))
            width, height = pil_image.size
            
            logger.info(f"[API] Batch processing {len(ids_list)} pakaian...")
            
            model = FluxKontextModel()
            results = []
            
            for i, pakaian_id in enumerate(ids_list):
                logger.info(f"[API] Batch {i+1}/{len(ids_list)}: {pakaian_id}")
                
                result = model.generate_pakaian_adat.remote(
                    image_bytes=image_bytes,
                    pakaian_id=pakaian_id,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    seed=i,  # Different seed for variation
                )
                
                results.append(result)
            
            logger.info(f"[API] Batch completed: {len(results)} images generated")
            
            return JSONResponse(
                content={
                    "status": "success",
                    "total_generated": len(results),
                    "results": results
                },
                status_code=200
            )
        
        except Exception as e:
            logger.error(f"[API] Batch error: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    # ========================================================================
    # HEALTH CHECK
    # ========================================================================
    
    @web_app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "service": "Pakaian Adat Generator",
            "model": "Flux-Kontext",
            "total_pakaian": len(PAKAIAN_ADAT_PROMPTS)
        }
    
    return web_app

# ============================================================================
# CLI ENTRYPOINT (for testing)
# ============================================================================

@app.local_entrypoint()
def main():
    """Test the model locally."""
    print("🎭 Pakaian Adat Generator - Test Mode")
    print("=" * 60)
    
    print(f"\n📋 Total pakaian adat available: {len(PAKAIAN_ADAT_PROMPTS)}")
    
    print("\n✅ Model configuration validated")
    print("\n🚀 Deploy with: modal deploy pakaian_adat_generator.py")
    print("\n📚 API Endpoints:")
    print("   • GET  /pakaian-adat/list      - List all pakaian")
    print("   • GET  /pakaian-adat/{id}      - Get pakaian detail")
    print("   • POST /generate               - Generate single pakaian")
    print("   • POST /generate-batch         - Generate multiple pakaian")
    print("   • GET  /health                 - Health check")