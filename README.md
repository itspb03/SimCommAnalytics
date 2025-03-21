## **SimCommAnalytics**  
Analysis of communication data in a group simulation setting to understand group dynamics. This project implements advanced data analytics and AI modeling to analyze human-to-human communication within simulated environments.    

---

## **Table of Contents**  
- [Overview](#overview)  
- [Requirements](#requirements)  
- [Installation & Setup](#installation--setup)  
- [Usage](#usage)  
  - [Processing Videos](#1-processing-videos)  
  - [Generating Visualizations](#2-generating-visualizations)  
- [Testing](#testing)
- [Output of Sample Data](#output-of-sample-data)
- [References](#references)
- [Future Improvements](#future-improvement)  

---

## **Overview**  
This project processes video files, extracts audio, segments it into **5-second chunks**, transcribes speech using **Whisper**, and performs **sentiment analysis** using a transformer model. The results are saved as CSV files and can be visualized using histograms and pie charts.

The project consists of two main scripts:  

- **`datahandling.py`** – Handles video downloading, audio extraction, transcription, and sentiment analysis.  
- **`data_plots.ipynb`** – Reads the processed CSV files and generates visualizations.  

---

## **Requirements**  
Ensure the following libraries are installed before running the scripts:  
```bash
pip install torch transformers whisper yt-dlp pandas requests bs4 pydub matplotlib numpy 
```

### **Required Libraries**  
| Library | Purpose |
|---------|---------|
| `torch` | Used for running deep learning models, including sentiment analysis. |
| `transformers` | Provides the pre-trained RoBERTa model for sentiment analysis. |
| `whisper` | Utilized for automatic speech recognition (ASR). |
| `yt-dlp` | Downloads video files from a given URL. |
| `pandas` | Handles tabular data and stores transcriptions in CSV format. |
| `requests` | Fetches HTML pages to extract video links. |
| `bs4` (BeautifulSoup) | Parses HTML to extract video links from webpages. |
| `pydub` | Handles audio file conversion and segmentation. |
| `matplotlib` | Generates plots for visualization. |
| `numpy` | Supports numerical computations for histogram binning. |

---

## **Installation & Setup**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/itspb03/SimCommAnalytics.git
cd SimCommAnalytics
```

### **2. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **3. Set Up Required Directories**  
The scripts automatically create the necessary directories (`videos/`, `audio/`, `output/`). Ensure that:  
- **Video files** are downloaded or placed in the `videos/` folder.  
- The **`output/` folder** stores processed CSV results.  

---

## **Usage**  

### **1. Processing Videos**  
Run `datahandling.py` to process a video file or URL.  

#### **Run the Script**  
```bash
python datahandling.py
```

#### **Example: Processing a Single Video**  
```python
main("path/to/local/video.mp4")
```

#### **Example: Processing a List of URLs**  
```python
main([
    "https://example.com/video1.mp4",
    "https://example.com/video2.mp4"
])
```

#### **Example: Processing All Videos in a Folder**  
```python
main("videos/")
```

The script:
1. Downloads videos from the provided source (if a URL).
2. Extracts audio and segments it into ≤5s chunks.
3. Transcribes each chunk using OpenAI’s Whisper model.
4. Performs sentiment analysis using the **cardiffnlp/twitter-roberta-base-sentiment** model.
5. Saves results in CSV format in the `output/` folder.




#### **Example CSV Output Format**
| Timestamp | Transcription | Sentiment |
|-----------|-------------|-----------|
| 0.00      | Hello world! | Neutral   |
| 5.00      | This is amazing. | Positive |

---

### **2. Generating Visualizations**  
Run `data_plots.ipynb` to visualize sentiment classification and word frequency histograms.  

 
### **This script reads the generated CSV file and produces:**  
  - A **histogram** of word counts per 5-second bucket (colored by sentiment).  
  - A **pie chart** showing the distribution of sentiment.  

#### **Run the script using:**  
```bash
jupyter notebook
```
- Open **`data_plots.ipynb`** and execute the cells.  

#### **How It Works:**  
1. Reads the CSV files from the `output/` directory.  
2. Extracts **timestamps, transcriptions, and sentiment classifications**.  
3. Generates:  
   - **Histogram:** Counts the words per 5-second window and colors bars based on sentiment.  
   - **Pie Chart:** Displays sentiment proportions.  

---

## **Testing**
Each component was tested using sample video files and URLs.

### **Unit Tests**
1. **Video Downloading** – Verified using multiple video URLs.
2. **Audio Extraction & Segmentation** – Tested on different formats (`mp4`, `mkv`, `webm`).
3. **Transcription Accuracy** – Compared Whisper output with expected transcriptions.
4. **Sentiment Analysis** – Manually verified sentiment labels for correctness.
5. **Visualization Outputs** – Checked histograms and pie charts for proper sentiment distribution.

### **Edge Cases**
- Empty transcriptions default to `Neutral`.
- Non-English speech might produce incorrect transcriptions.
- Poor audio quality affects transcription accuracy.

---

## **Output of Sample Data**  

Below is an example of the output generated from a sample video processed through our pipeline.  

### **Sample Video**  
[[Click here to view the sample video](https://alabama.app.box.com/s/vnwa6b4qi42ts39exv8jej6vzabdu78k)]

### **Generated Visualizations**  

#### **1. Histogram: Word Count Per 5-Second Interval**  
*(This histogram represents the frequency of words spoken in each 5-second chunk, color-coded based on sentiment classification.)*  
![image](https://github.com/user-attachments/assets/b7f8a797-4665-4e65-b82b-b7a3c84499f5)]   

#### **2. Pie Chart: Sentiment Distribution**  
*(This pie chart visualizes the overall sentiment distribution across the transcription.)*  
![image](https://github.com/user-attachments/assets/d07e9ec2-cd7f-48d3-9191-22bd2d2d10f3)] 

---





## **References**  

### **1. Whisper Model (Automatic Speech Recognition)**  
- **GitHub**: [OpenAI Whisper](https://github.com/openai/whisper)  
- **Paper**: [Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)  

### **2. Sentiment Analysis Model**  
- **Model**: [Cardiff NLP RoBERTa](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)  
- **Paper**: [TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification](https://arxiv.org/abs/2010.12421)  

### **3. YouTube-DLP for Video Downloading**  
- **Repository**: [yt-dlp](https://github.com/yt-dlp/yt-dlp)  

### **4. Pydub for Audio Processing**  
- **Documentation**: [Pydub](https://github.com/jiaaro/pydub)  



## **Future Improvements**
- Add support for multiple languages.
- Improve segmentation logic for more accurate timestamps.
- Enhance UI for better visualization.

---



















