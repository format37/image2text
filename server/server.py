from flask import Flask, request, jsonify
import os
import uuid
import logging
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Starting server")

app = Flask(__name__)

logger.info("Loading model")
"""pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            revision="fp16",
            use_auth_token= os.environ.get("HUGGINGFACE_USE_AUTH_TOKEN", False),
            cache_dir='/app/data/models/'
            )"""
# model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
model_id = 'nlpconnect/vit-gpt2-image-captioning'
model = VisionEncoderDecoderModel.from_pretrained(
    model_id
    )
logger.info("Saving model")
# save model to disk
model.save_pretrained("data/vit-gpt2-image-captioning")

logger.info("Loading extractor")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
logger.info("Loading tokinizer")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

logger.info("Sending to device")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}


def predict_step(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


@app.route("/test")
def call_test():
    return "get ok"


@app.route("/request", methods=["POST"])
def call_request():
    logger.info("request")
    if request.method == "POST":
        # get file from request
        file = request.files["file"]
        # generate unique file name
        filepath = 'data/'+str(uuid.uuid4()) + ".jpg"
        # save file to local folder
        file.save(filepath)
        description = str(predict_step([filepath]))
        # return file name
        # return jsonify({"filename": filename})
        logger.info("Description: " + description)
        return jsonify({"description": description})
    return jsonify({"description": "nothing"})


if __name__ == "__main__":
    # Only for debugging while developing
    app.run(host='0.0.0.0', debug=False, port=os.environ.get('PORT', 10001))
