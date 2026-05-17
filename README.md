
# Fire Detection

 Development of an intelligent fire detection system combining vision and contextual information


 
<div align="center">
  <a href="https://github.com/user-attachments/assets/521ab2d3-1821-480c-aa3e-835d267afd67">
    <img width="1265" height="1244" alt="system architecture - research paper" src="https://github.com/user-attachments/assets/521ab2d3-1821-480c-aa3e-835d267afd67" />
  </a>
  <p><i> System Architecture (Click to enlarge)</i></p>
</div>

<br><br><br>





## Dangerous-fire Web User Interface

The browser UI lives in the `software` folder.

```powershell
cd software
python -m pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000/
```

Use a Chromium-based browser and choose individual image files or an image folder. The app filters supported image types and tests those images with YOLO11 + CLIP.
<br><br><br>

## Licenses & Attributions

This project uses components from the following third-party open-source projects:

1. **Ultralytics YOLO11** - Used for initial fire and smoke object detection. Distributed under the [GNU AGPL-3.0 License](https://github.com/ultralytics/ultralytics/blob/main/LICENSE). Accordingly, this entire repository is open-sourced under the AGPL-3.0 standard.
2. **OpenAI CLIP (ViT-B/32)** - Used for cross-modal image-text verification. Distributed under the permissive [MIT License](https://github.com/openai/CLIP/blob/main/LICENSE).
3. **Fine-Tuned Weights (`leeyunjai/yolo11-firedetect`)** - The fine-tuned weights utilized in our pipelines are hosted on [Hugging Face](https://huggingface.co/leeyunjai/yolo11-firedetect).