RENDERED_IMAGES_DIR="/data/tamper/teddybear/images/"      # path to rendered images
GS_MODEL_PATH="/data/gs_model/bicycle/teddybear/"        # path to the 3DGS model
SAFIRE_WEIGHT="safire.pth"                        
SAFIRE_OUTPUT="/data/SAFIRE/bicycle/teddybear/"

cd SAFIRE
python infer_binary.py \
    --resume="${SAFIRE_WEIGHT}" \
    --input_path="${RENDERED_IMAGES_DIR}" \
    --save_path="${SAFIRE_OUTPUT}"

cd ..
python tamper_loc.py \
    -m "${GS_MODEL_PATH}" \
    --out_dir "${SAFIRE_OUTPUT}"