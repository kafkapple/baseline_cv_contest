import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import glob
from pathlib import Path
from tqdm import tqdm
import json
import datetime
import pandas as pd
# -------------------------
# 1. Embedding Model Class (변동 없음)
# -------------------------
class EmbeddingModel:
    def __init__(self, model_name="resnet50"):
        if model_name == "resnet50":
            self.model = models.resnet50(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer
        elif model_name == "mobilenet_v2":
            self.model = models.mobilenet_v2(pretrained=True)
            self.feature_extractor = nn.Sequential(*list(self.model.children())[:-1])  # Remove FC layer
        else:
            raise ValueError("Unsupported model name. Choose 'resnet50' or 'mobilenet_v2'.")
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print(f"EmbeddingModel initialized with {model_name} model")
    
    def extract_embedding(self, image_path):
        img = Image.open(image_path).convert('RGB')
        input_tensor = self.preprocess(img).unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            embedding = self.feature_extractor(input_tensor).squeeze().numpy()
        return embedding.ravel()  # Flatten to 1D

# -------------------------
# 2. Clustering Method Class (변동 없음)
# -------------------------
class ClusteringMethod:
    def __init__(self, method_name="kmeans", **kwargs):
        if method_name == "kmeans":
            self.method = KMeans(**kwargs)
        else:
            raise ValueError("Unsupported clustering method. Choose 'kmeans'.")
        print(f"ClusteringMethod initialized with {method_name} method")
    
    def fit_predict(self, data):
        self.labels = self.method.fit_predict(data)
        return self.labels
    
    def get_centroids(self, data):
        if hasattr(self.method, "cluster_centers_"):  # For KMeans
            return self.method.cluster_centers_
        else:
            raise ValueError("Centroids not available for this clustering method.")

# -------------------------
# 3. Main Pipeline with Outlier Detection
# -------------------------
class EmbeddingClusteringPipeline:
    def __init__(self, image_paths, embedding_model, clustering_method, n_images=3, output_dir="results"):
        self.image_paths = image_paths
        self.embedding_model = embedding_model
        self.clustering_method = clustering_method
        
        # 타임스탬프 기반 하위 폴더 생성
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(output_dir, timestamp)
        self.embeddings_dir = os.path.join(self.output_dir, "embeddings")
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.embeddings_dir, exist_ok=True)
        
        self.n_images = n_images
    def extract_and_save_embeddings(self):
        """이미지 임베딩을 추출하고 저장"""
        print("Extracting and saving embeddings...")
        embeddings_dict = {}
        
        for path in tqdm(self.image_paths):
            img_name = os.path.basename(path)
            embedding_path = os.path.join(self.embeddings_dir, f"{os.path.splitext(img_name)[0]}.npy")
            
            # 이미 계산된 임베딩이 있는지 ��인
            if os.path.exists(embedding_path):
                embedding = np.load(embedding_path)
            else:
                # 새로운 임베딩 추출 및 저장
                embedding = self.embedding_model.extract_embedding(path)
                np.save(embedding_path, embedding)
            
            embeddings_dict[path] = embedding
        
        self.embeddings = np.array(list(embeddings_dict.values()))
        return embeddings_dict
    
    def process_and_visualize(self):
        # 1. 임베딩 추출 및 저장
        print("1. Extracting and saving embeddings...")
        embeddings_dict = self.extract_and_save_embeddings()
        
        # 2. 2차원으로 차원 축소
        print("2. Reducing dimensions to 2D...")
        pca = PCA(n_components=2)
        self.viz_data = pca.fit_transform(self.embeddings)
        self.viz_data = (self.viz_data - self.viz_data.min(axis=0)) / (self.viz_data.max(axis=0) - self.viz_data.min(axis=0))
        
        # 3. 클러스터링
        print("3. Performing clustering...")
        self.labels = self.clustering_method.fit_predict(self.viz_data)
        
        # 4. 대표 샘플과 아웃라이어 찾기
        print("4. Finding representative and outlier samples...")
        self.find_representative_and_outlier_samples()
        
        # 5. 결과 저장
        print("5. Saving results...")
        self.save_results()
        
        # 5.1 HTML 보고서 저장
        print("5.1 Saving HTML report...")
        self.save_html_report()  # 각 클러스터별 대표 샘플 n개 저장
        
        # 6. 시각화
        print("6. Visualizing results...")
        self.visualize_clusters()
    
    def find_representative_and_outlier_samples(self):
        """각 클러스터의 대표 샘플과 아웃라이어 찾기"""
        self.representative_samples = []
        self.outliers = []
        
        for label in range(len(np.unique(self.labels))):
            mask = self.labels == label
            cluster_points = self.viz_data[mask]
            cluster_paths = np.array(self.image_paths)[mask]
            
            # 클러스터 중심 계산
            center = cluster_points.mean(axis=0)
            
            # 중심까지의 거리 계산
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            # 대표 샘플 (중심에 가장 가까운 샘플)
            representative_idx = np.argmin(distances)
            self.representative_samples.append(cluster_paths[representative_idx])
            
            # 아웃라이어 (중심에서 가장 먼 샘플)
            outlier_idx = np.argmax(distances)
            self.outliers.append(cluster_paths[outlier_idx])
    
    def save_results(self):
        """클러스터링 결과 저장"""
        # 클러스터 할당 저장 (numpy.int32를 int로 변환)
        cluster_assignments = {
            str(path): int(label)  # path를 문자열로, label을 int로 변환
            for path, label in zip(self.image_paths, self.labels)
        }
        
        with open(os.path.join(self.output_dir, "cluster_assignments.json"), 'w') as f:
            json.dump(cluster_assignments, f, indent=2)
        
        # 대표 샘플 저장
        with open(os.path.join(self.output_dir, "representative_samples.txt"), "w") as f:
            f.write("\n".join(str(path) for path in self.representative_samples))
        
        # 아웃라이어 저장
        with open(os.path.join(self.output_dir, "outlier_samples.txt"), "w") as f:
            f.write("\n".join(str(path) for path in self.outliers))
        
        # 임베딩 메타데이터 저장
        metadata = {
            "num_clusters": int(len(np.unique(self.labels))),  # numpy.int32를 int로 변환
            "num_samples": int(len(self.image_paths)),
            "embedding_dim": int(self.embeddings.shape[1]),
            "model_name": self.embedding_model.model.__class__.__name__
        }
        
        with open(os.path.join(self.output_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def visualize_clusters(self):
        """클러스터 시각화"""
        plt.figure(figsize=(12, 8))
        
        # 클러스터별 색상 지정
        colors = plt.cm.rainbow(np.linspace(0, 1, len(np.unique(self.labels))))
        
        # 먼든 클러스터 포인트를 그립니다
        unique_labels = np.unique(self.labels)
        for label, color in zip(unique_labels, colors):
            cluster_points = self.viz_data[self.labels == label]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                       color=color,
                       label=f"Cluster {label}", 
                       alpha=0.7,
                       s=100,  # 포인트 크기 더 증가
                       marker='o',
                       zorder=2)
        
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title("Cluster Visualization")
        
        # 축 레이블 추가
        plt.xlabel("First Principal Component")
        plt.ylabel("Second Principal Component")
        
        # 여백 조정
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, "cluster_visualization.png"), 
                    bbox_inches='tight', 
                    dpi=300)
        
        plt.show(block=False)
        plt.pause(5)
        plt.close()
    
    def save_html_report(self):
        """각 클러스터의 대표 샘플을 HTML 보고서로 저장"""
        # 이미지를 저장할 디렉토리 생성
        images_dir = os.path.join(self.output_dir, "images")
        os.makedirs(images_dir, exist_ok=True)
        
        html_content = """
        <html>
        <head>
            <title>Cluster Report</title>
            <style>
                .cluster-container { margin-bottom: 30px; }
                .image-container { display: inline-block; margin: 10px; text-align: center; }
                img { margin-bottom: 5px; }
            </style>
        </head>
        <body>
        <h1>Cluster Report</h1>
        """
        
        for label in range(len(np.unique(self.labels))):
            html_content += f'<div class="cluster-container"><h2>Cluster {label}</h2>'
            mask = self.labels == label
            cluster_paths = np.array(self.image_paths)[mask]
            cluster_points = self.viz_data[mask]
            
            # 중심까지의 거리 계산
            center = cluster_points.mean(axis=0)
            distances = np.linalg.norm(cluster_points - center, axis=1)
            
            # 거리 기준으로 정렬하여 상위 n개 선택
            sorted_indices = np.argsort(distances)[:self.n_images]
            representative_samples = cluster_paths[sorted_indices]
            
            for i, sample in enumerate(representative_samples):
                # 이미지 파일 복사
                img_filename = f"cluster_{label}_sample_{i}_{os.path.basename(sample)}"
                img_dest_path = os.path.join(images_dir, img_filename)
                try:
                    import shutil
                    shutil.copy2(sample, img_dest_path)
                    
                    # HTML에 상대 경로로 이미지 추가 (./images/로 시작하는 상대 경로 사용)
                    relative_path = f"./images/{img_filename}"
                    html_content += f"""
                    <div class="image-container">
                        <img src="{relative_path}" width="200">
                        <p>File: {os.path.basename(sample)}</p>
                        <p>Distance from center: {distances[sorted_indices[i]]:.4f}</p>
                    </div>
                    """
                except Exception as e:
                    print(f"Error copying image {sample}: {str(e)}")
            
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        with open(os.path.join(self.output_dir, "cluster_report.html"), "w") as f:
            f.write(html_content)

def save_embeddings(model, data_loader, device, cfg):
    """모델의 중간 레이어 임베딩을 추출하고 저장"""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Extracting embeddings"):
            images = images.to(device)
            # 예시: ResNet의 경우 avgpool 이전 레이어의 출력을 사용
            features = model.get_features(images)  # 모델에 get_features 메서드 필요
            embeddings.append(features.cpu().numpy())
            labels.extend(targets.numpy())
    
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.array(labels)
    
    # 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = output_dir / f"embeddings_{timestamp}.npz"
    np.savez(
        save_path,
        embeddings=embeddings,
        labels=labels
    )
    
    print(f"Embeddings saved to {save_path}")
    return save_path

# -------------------------
# 4. Run the Pipeline
# -------------------------

class DatasetVisualizer:
    def __init__(self, csv_path: str, img_dir: str, output_dir: str = "results/dataset_viz"):
        """
        Args:
            csv_path: target 정보가 포함된 CSV 파일 경로
            img_dir: 이미지 디렉토리 경로
            output_dir: 결과물 저장 디렉토리
        """
        self.df = pd.read_csv(csv_path)
        self.img_dir = Path(img_dir)
        self.base_output_dir = Path(output_dir)
        
        # timestamp 기반 run id 생성 및 폴더 생성
        self.run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.base_output_dir / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 클래스별 이미지 경로 정리 (클래스 ID 순서대로)
        self.class_images = {}
        for class_id in sorted(self.df['target'].unique()):  # 정렬된 클래스 ID
            class_df = self.df[self.df['target'] == class_id]
            self.class_images[class_id] = [
                self.img_dir / img_id for img_id in class_df['ID']
            ]
    
    def create_class_report(self, n_samples: int = 10, seed: int = 42):
        """각 클래스별 랜덤 샘플링된 이미지로 HTML 리포트 생성"""
        np.random.seed(seed)
        
        # 이미지 저장 디렉토리 (run id 하위에 생성)
        images_dir = self.output_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        html_content = f"""
        <html>
        <head>
            <title>Dataset Class Report - {self.run_id}</title>
            <style>
                .class-container {{ margin-bottom: 30px; }}
                .image-container {{ display: inline-block; margin: 10px; text-align: center; }}
                img {{ max-width: 200px; margin-bottom: 5px; }}
                .class-info {{ margin-bottom: 10px; }}
                .run-info {{ margin-bottom: 20px; color: #666; }}
            </style>
        </head>
        <body>
        <h1>Dataset Class Report</h1>
        <div class="run-info">
            Run ID: {self.run_id}<br>
            Total Images: {len(self.df)}<br>
            Number of Classes: {len(self.class_images)}
        </div>
        """
        
        # 전체 클래스 분포 정보 추가 (정렬된 순서로)
        class_dist = self.df['target'].value_counts().sort_index()  # 클래스 ID 순서대로 정렬
        html_content += "<h2>Class Distribution</h2>"
        html_content += "<table border='1'><tr><th>Class</th><th>Count</th><th>Ratio(%)</th></tr>"
        for cls in sorted(class_dist.index):  # 정렬된 순서로 출력
            count = class_dist[cls]
            ratio = count / len(self.df) * 100
            html_content += f"<tr><td>{cls}</td><td>{count}</td><td>{ratio:.1f}%</td></tr>"
        html_content += "</table><br>"
        
        # 각 클래스별 샘플 이미지 (정렬된 순서로)
        for class_id in sorted(self.class_images.keys()):  # 클래스 ID 순서대로
            img_paths = self.class_images[class_id]
            html_content += f'<div class="class-container">'
            html_content += f'<h2>Class {class_id}</h2>'
            
            # 클래스 정보
            html_content += f'<div class="class-info">'
            html_content += f'Total images: {len(img_paths)}<br>'
            html_content += f'Ratio: {len(img_paths)/len(self.df)*100:.1f}%'
            html_content += '</div>'
            
            # 랜덤 샘플링
            selected_paths = np.random.choice(img_paths, 
                                           min(n_samples, len(img_paths)), 
                                           replace=False)
            
            # 이미지 추가
            for i, img_path in enumerate(selected_paths):
                img_filename = f"class_{class_id}_sample_{i}_{img_path.name}"
                img_dest_path = images_dir / img_filename
                
                try:
                    import shutil
                    shutil.copy2(img_path, img_dest_path)
                    
                    html_content += f"""
                    <div class="image-container">
                        <img src="./images/{img_filename}">
                        <p>File: {img_path.name}</p>
                    </div>
                    """
                except Exception as e:
                    print(f"Error copying image {img_path}: {str(e)}")
            
            html_content += "</div>"
        
        html_content += "</body></html>"
        
        # HTML 파일 저장
        with open(self.output_dir / "class_report.html", "w") as f:
            f.write(html_content)
        
        print(f"Report saved to {self.output_dir}/class_report.html")
        print(f"Run ID: {self.run_id}")

# 사용 예시
if __name__ == "__main__":
    # Example image paths
    test_path = Path(__file__).parent / "data/raw/train/"
    #test_path = "/home/joon/dataset/cv/test/images"
    print(f"\n===test_path: {test_path}")
    image_paths = glob.glob(f"{test_path}/*.jpg")
    print(f"image len: {len(image_paths)}")
    # Initialize Embedding Model (ResNet50)
    # embedding_model = EmbeddingModel(model_name="mobilenet_v2")
    
    # # Initialize Clustering Method (KMeans)
    # clustering_method = ClusteringMethod(method_name="kmeans", n_clusters=17, random_state=42)
    
    # # Run the Pipeline
    # pipeline = EmbeddingClusteringPipeline(
    #     image_paths=image_paths,
    #     embedding_model=embedding_model,
    #     clustering_method=clustering_method,
    #     n_images=10,
    #     output_dir="outputs/embedding"
    # )
    
    # pipeline.process_and_visualize()
    
    # 데이터셋 시각화
    visualizer = DatasetVisualizer(
        csv_path="data/raw/train.csv",
        img_dir="data/raw/train",
        output_dir="outputs/dataset_viz"
    )
    visualizer.create_class_report(n_samples=10)
