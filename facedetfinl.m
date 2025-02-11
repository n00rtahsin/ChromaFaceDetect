function main_face_detection()
% Main function that handles image selection and initiates face detection process
[filename, pathname] = uigetfile({'*.jpg;*.png;*.bmp', 'Image Files (*.jpg,*.png,*.bmp)'});% Open file selection dialog
if filename == 0
    return;
end
% Reading the selected image file
img = imread(fullfile(pathname, filename));
% Converting image to double precision for processing
img_double = im2double(img); % Converts image to double precision, scaling pixel values to [0, 1]
% Processing image and detecting faces
[faces, bbox] = detect_faces(img_double);
% Displaying the results with detected faces
display_results(img, bbox);
end

function [faces, bbox] = detect_faces(img)
% Function to detect faces in the input image using skin color detection

% Converting RGB image to YCbCr color space for better skin detection
ycbcr = rgb2ycbcr(img); % Converts the image from RGB to YCbCr color space to isolate luminance and chrominance components

% Extracting chrominance components
Cb = ycbcr(:,:,2); % Extracting the Cb (blue-difference chroma) component
Cr = ycbcr(:,:,3); % Extracting the Cr (red-difference chroma) component

% Applying skin color thresholds based on research papers
skin_mask = (Cb >= 0.4 & Cb <= 0.6 & Cr >= 0.54 & Cr <= 0.68); % Logical mask isolating potential skin regions

% Cleaning up the mask using morphological operations
skin_mask = imopen(skin_mask, strel('disk', 5)); % Performs morphological opening to remove small noise
skin_mask = imclose(skin_mask, strel('disk', 5)); % Performs morphological closing to fill gaps
skin_mask = bwareaopen(skin_mask, 1000); % Removes connected regions smaller than 1000 pixels

% Labeling connected regions and analyze their properties
[labeled, num_regions] = bwlabel(skin_mask); % Labels connected regions in the mask
stats = regionprops(labeled, 'BoundingBox', 'Area', 'Centroid', 'Eccentricity'); % Computes region properties like bounding box and shape

% Initializing output arrays
bbox = []; % Array to store bounding boxes of detected faces
faces = []; % Array to store cropped face regions

% Analyzing each detected region
for i = 1:num_regions
    current_bbox = stats(i).BoundingBox; % Retrieves the bounding box of the current region

    % Calculate aspect ratio and check face criteria
    aspect_ratio = current_bbox(3) / current_bbox(4); % Aspect ratio: width divided by height
    eccentricity = stats(i).Eccentricity; % Eccentricity measures how elongated a region is

    if aspect_ratio >= 0.5 && aspect_ratio <= 1.5 && ...
       eccentricity < 0.85 && ...
       stats(i).Area > 2000 % Ensures the region meets size and shape criteria

        % Extract and verify facial features
        region = imcrop(img, current_bbox); % Crops the region based on the bounding box
        if has_facial_features(region) % Checks if the region contains facial features
            bbox = [bbox; current_bbox]; % Adds the bounding box to the output array
            faces = [faces; region]; % Adds the cropped face to the output array
        end
    end
end
end

function has_features = has_facial_features(face_region)
% Function to verify the presence of facial features in a detected region

% Converting region to grayscale
gray_face = rgb2gray(face_region); % Converts the RGB face region to grayscale

% Detecting edges using Canny edge detector
edges = edge(gray_face, 'Canny'); % Detects edges using the Canny method

% Calculating vertical projection for feature detection
vertical_proj = sum(edges, 2); % Sums edge pixels vertically to detect features like eyes and mouth

% Normalizing the projection
vertical_proj = vertical_proj / max(vertical_proj); % Normalizes the projection values

% Detecting peaks corresponding to facial features
[peaks, locs] = findpeaks(vertical_proj, 'MinPeakHeight', 0.3, 'MinPeakDistance', size(gray_face, 1)/6); % Finds peaks representing facial features

% Verifying presence of sufficient facial features
has_features = length(peaks) >= 3; % Checks if there are at least 3 peaks, indicating potential facial features
end

function display_results(img, bbox)
% Function to display detection results

figure('Name', 'Face Detection Results', 'NumberTitle', 'off'); % Creates a new figure window with a title

imshow(img); % Displays the original image
hold on; % Retains the image while adding bounding boxes

% Drawing bounding boxes around detected faces
for i = 1:size(bbox, 1)
    rectangle('Position', bbox(i,:), 'EdgeColor', 'g', 'LineWidth', 2); % Draws green rectangles around detected faces
end
title(sprintf('Detected %d faces', size(bbox, 1))); % Adds a title displaying the number of detected faces
hold off; % Releases the hold on the figure
end