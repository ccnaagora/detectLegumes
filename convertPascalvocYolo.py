import os
import glob
import xml.etree.ElementTree as ET

def convert_pascalvoc_to_yolo(xml_dir, output_dir, classes_file):
    # Lire les noms de classes
    with open(classes_file, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]

    # Créer le dossier de sortie s'il n'existe pas
    os.makedirs(output_dir, exist_ok=True)

    # Parcourir tous les fichiers XML
    for xml_file in glob.glob(os.path.join(xml_dir, '*.xml')):
        # Lire le XML
        tree = ET.parse(xml_file)
        root = tree.getroot()

        # Récupérer les infos de l'image
        filename = root.find('filename').text
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)

        # Créer le fichier YOLO correspondant
        yolo_filename = os.path.join(output_dir, os.path.splitext(filename)[0] + '.txt')
        with open(yolo_filename, 'w') as yolo_file:
            for obj in root.findall('object'):
                # Récupérer les infos de l'objet
                class_name = obj.find('name').text
                class_id = class_names.index(class_name)

                # Récupérer les coordonnées de la boîte
                xmin = int(obj.find('bndbox/xmin').text)
                ymin = int(obj.find('bndbox/ymin').text)
                xmax = int(obj.find('bndbox/xmax').text)
                ymax = int(obj.find('bndbox/ymax').text)

                # Calculer les coordonnées normalisées YOLO
                x_center = (xmin + xmax) / 2 / width
                y_center = (ymin + ymax) / 2 / height
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height

                # Écrire dans le fichier YOLO
                yolo_file.write(f"{class_id} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n")

# Exemple d'utilisation
xml_dir = "mondataset/train/annotations"
output_dir = "yolo/train"
classes_file = "classes.txt"
convert_pascalvoc_to_yolo(xml_dir, output_dir, classes_file)
