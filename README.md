# Custom_Render_Img

### Prerequisites
- A trained nerfstudio model
- Place custom_show_img.py under nerfstudio/nerfstudio/scripts/ folder

### Usage
```bash
# Go to the root folder of nerfstudio
cd nerfstudio

# Execute the script
# arg1: Relative path to the config file
# arg2: Folder for the output images
python nerfstudio/scripts/custom_show_img.py outputs/PATH/TO/config.yml custom_output
```
