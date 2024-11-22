import os
import numpy as np
import pydicom
import dicom2nifti
import cv2
import matplotlib.pyplot as plt

pacientes = {}
archivos = {}


class Paciente:
    def _init_(self, identificacion, edad, nifti_path):
        self._id = identificacion  # El atributo _id es privado
        self._edad = edad
        self._nifti_path = nifti_path  # Ruta del archivo NIfTI asociado

    def get_id(self):
        return self._id

    def info(self):
        # Método para acceder a la información del paciente
        return f"Información del paciente: ID: {self.get_id()}, Edad: {self._edad}, Imagen NIfTI: {self._nifti_path}"


def leer_archivos(carpeta):
    def leer_archivos(carpeta):
    """
    Lee una carpeta o archivo DICOM, extrae la información del paciente y convierte a NIfTI.
    """
    carpeta = os.path.normpath(carpeta)  # Normalizar ruta
    print(f"Ruta proporcionada: {carpeta}")
    
    # Solicitar al usuario que ingrese una clave para identificar este archivo
    clave = input("Ingrese una clave para identificar este archivo: ")

    try:
        if os.path.isdir(carpeta):
            archivos_dicom = [os.path.join(carpeta, f) for f in os.listdir(carpeta) if f.lower().endswith(".dcm")]
            if not archivos_dicom:
                raise FileNotFoundError(f"No se encontraron archivos DICOM en la carpeta: {carpeta}")
        elif os.path.isfile(carpeta) and carpeta.lower().endswith(".dcm"):
            archivos_dicom = [carpeta]
        else:
            raise ValueError("La ruta proporcionada no es válida. Debe ser un archivo o carpeta de archivos DICOM.")

        # Leer el primer archivo para extraer la información del paciente
        ds = pydicom.dcmread(archivos_dicom[0])
        id_paciente = getattr(ds, 'PatientID', 'Desconocido')
        edad_paciente = getattr(ds, 'PatientAge', 'Desconocida')

        # Crear directorio 'salida' si no existe
        if not os.path.exists("salida"):
            os.makedirs("salida")
        
        # Convertir a NIfTI
        dicom2nifti.convert_directory(carpeta, "salida", reorient=True)
        
        # Obtener el archivo NIfTI generado
        nifti_archivos = os.listdir("salida")
        if not nifti_archivos:
            raise FileNotFoundError("No se generaron archivos NIfTI.")
        ruta_nifti = os.path.join("salida", nifti_archivos[0])

        # Crear el objeto Paciente usando la clave, ID, edad y la imagen NIfTI
        paciente = Paciente(id_paciente, edad_paciente, ruta_nifti)
        paciente.clave = clave  # Asignar la clave al paciente

        # Guardar al paciente con la clave en el diccionario 'pacientes'
        pacientes[clave] = paciente  
        archivos[clave] = archivos_dicom  # Guardar archivos DICOM con la clave

        print(f"Paciente agregado: ID={id_paciente}, Edad={edad_paciente}, NIfTI={ruta_nifti}, Clave={clave}")
        return paciente
    except Exception as e:
        print(f"Error al cargar el paciente: {e}")

    
def ajustar_escala(pixel_data, dicom_dataset):
    """Ajusta la escala de los datos de píxeles para asegurar que estén en formato adecuado para el almacenamiento."""
    # Obtener la información de escalado
    rescale_slope = dicom_dataset.get('RescaleSlope', 1)
    rescale_intercept = dicom_dataset.get('RescaleIntercept', 0)
    
    # Convertir los datos de píxeles a un numpy array
    pixel_array = np.frombuffer(pixel_data, dtype=np.uint16)  # Usamos uint16 como ejemplo; ajusta según el caso

    # Aplicar la escala
    pixel_array = pixel_array * rescale_slope + rescale_intercept

    # Clipping para evitar valores fuera del rango uint16
    pixel_array = np.clip(pixel_array, 0, 65535)  # Asegurarse de que no exceda el rango para uint16

    # Regresar los datos ajustados como bytes
    return pixel_array.tobytes()

def convertir_dicom_a_nifti(dicom_folder, nifti_output):
    try:
        dicom2nifti.convert_directory(dicom_folder, nifti_output)
        print(f"Conversión de DICOM a NIfTI completada: {nifti_output}")
    except Exception as e:
        print(f"Error al convertir DICOM a NIfTI: {e}")

def rotacion(imagen, angulo, salida="rotada.png"):
    if angulo not in [90, 180, 270]:
        raise ValueError("El ángulo debe ser 90, 180 o 270 grados.")

    # Verificar si la imagen es válida
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen correctamente.")

    # Rotación usando OpenCV
    if angulo == 90:
        im_rotada = cv2.rotate(imagen, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif angulo == 180:
        im_rotada = cv2.rotate(imagen, cv2.ROTATE_180)
    elif angulo == 270:
        im_rotada = cv2.rotate(imagen, cv2.ROTATE_90_CLOCKWISE)

    # Mostrar imágenes
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(imagen, cmap="gray")
    plt.title("Imagen Original")
    plt.subplot(1, 2, 2)
    plt.imshow(im_rotada, cmap="gray")
    plt.title(f"Imagen Rotada {angulo}°")
    plt.show()

    # Guardar imagen rotada
    cv2.imwrite(salida, im_rotada)
    return im_rotada

def binarizacion_transformacion(imagen, umbral, kernel, salida="binarizada.png"):
    # Verificar si la imagen es válida
    if imagen is None:
        raise ValueError("No se pudo cargar la imagen correctamente.")

    # Binarización
    _, im_binarizada = cv2.threshold(imagen, umbral, 255, cv2.THRESH_BINARY)

    # Transformación morfológica
    size_kernel = np.ones((kernel, kernel), np.uint8)
    im_transformada = cv2.morphologyEx(im_binarizada, cv2.MORPH_CLOSE, size_kernel)

    # Añadir texto
    texto = f"Imagen binarizada\nUmbral: {umbral}\nKernel: {kernel}x{kernel}"
    cv2.putText(im_transformada, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1, cv2.LINE_AA)

    # Mostrar y guardar imagen transformada
    plt.imshow(im_transformada, cmap="gray")
    plt.title("Imagen Transformada con Texto")
    plt.show()
    cv2.imwrite(salida, im_transformada)
    return im_transformada