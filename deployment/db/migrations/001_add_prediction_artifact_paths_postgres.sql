ALTER TABLE prediction_image_links
    ADD COLUMN IF NOT EXISTS source_path TEXT,
    ADD COLUMN IF NOT EXISTS mask_path TEXT,
    ADD COLUMN IF NOT EXISTS roi_path TEXT,
    ADD COLUMN IF NOT EXISTS overlay_path TEXT;
