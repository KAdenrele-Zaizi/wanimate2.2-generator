docker build -t wan2_2_auto . && docker run --rm \
                                            --gpus "device=0" \
                                            --ipc=host \
                                            -v /mnt/data3/WAN/models:/workspace/Wan2.2/Wan2.2-TI2V-5B \
                                            -v /mnt/data/Shire_Opal/videos/deepfake_detection/Wan:/workspace/Wan2.2/outputs \
                                            -v /mnt/data/Shire_Opal/videos/deepfake_detection:/workspace/outputs_dir \
                                            wan2_2_auto