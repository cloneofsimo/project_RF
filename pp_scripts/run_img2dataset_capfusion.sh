img2dataset --url_list /home/host/simo/capfusion --input_format "parquet"\
         --url_col "image_url" --caption_col "capsfusion" --output_format webdataset\
           --output_folder /home/host/simo/capfusion_256 --processes_count 32 --thread_count 256 --image_size 256\
            --resize_only_if_bigger=True --resize_mode="center_crop" --skip_reencode=True --min_image_size 256\
             --save_additional_columns '["laion_2b","laion_coco"]' --enable_wandb False