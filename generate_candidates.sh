python3 src/blink/scripts/generate_candidates.py --path_to_model_config "models/biencoder_wiki_large.json" --path_to_model "models/biencoder_wiki_large.bin" --entity_dict_path "models/entities_to_add.jsonl" --saved_cand_ids "models/new_entities.json" --encoding_save_file_dir "models" --chunk_end 3
# python3 src/blink/scripts/generate_candidates_clearml.py --input_dataset_name "BLINK_models" --path_to_model_config "biencoder_wiki_large.json" --path_to_model "biencoder_wiki_large.bin" --entity_dict_path "new_entity.jsonl" --saved_cand_ids "new_entities_ids.json" --encoding_save_file_dir "models"
