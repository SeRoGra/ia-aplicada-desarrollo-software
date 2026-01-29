# MVP - Clasificación de textos legales por tema
# Requisitos: pip install scikit-learn pandas

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix

# 1) Dataset sintético.
data = [
    # CIVIL (6)
    ("Demanda por incumplimiento de contrato de arrendamiento y restitución del inmueble", "civil"),
    ("Proceso de sucesión intestada y reparto de bienes entre herederos", "civil"),
    ("Reclamación por incumplimiento de contrato de compraventa y devolución del dinero", "civil"),
    ("Solicitud de indemnización por daños y perjuicios derivados de un accidente en propiedad privada", "civil"),
    ("Demanda por responsabilidad civil extracontractual por perjuicios ocasionados a un tercero", "civil"),
    ("Acción para el cobro de una deuda y reconocimiento de intereses moratorios", "civil"),

    # PENAL (6)
    ("Denuncia por hurto calificado y agravado con violencia", "penal"),
    ("Investigación por lesiones personales y solicitud de medidas de protección", "penal"),
    ("Denuncia por estafa mediante suplantación y transacciones fraudulentas", "penal"),
    ("Proceso penal por amenazas y constreñimiento contra la víctima", "penal"),
    ("Querella por daño en bien ajeno y destrucción de propiedad", "penal"),
    ("Solicitud de medida de aseguramiento dentro de investigación por fraude", "penal"),

    # LABORAL (6)
    ("Reclamación por despido sin justa causa y pago de indemnización", "laboral"),
    ("Solicitud de reconocimiento y pago de horas extras y recargos nocturnos", "laboral"),
    ("Demanda por no pago de salarios y prestaciones sociales adeudadas", "laboral"),
    ("Reclamación por accidente laboral y reconocimiento de incapacidades", "laboral"),
    ("Solicitud de reintegro laboral por despido en condición de debilidad manifiesta", "laboral"),
    ("Queja por acoso laboral y solicitud de investigación interna", "laboral"),

    # FAMILIA (6)
    ("Proceso de custodia y alimentos para menor de edad", "familia"),
    ("Solicitud de divorcio y regulación de visitas", "familia"),
    ("Demanda de impugnación de paternidad y pruebas de filiación", "familia"),
    ("Fijación de cuota alimentaria y medidas provisionales", "familia"),
    ("Proceso de adopción y estudio de idoneidad familiar", "familia"),
    ("Solicitud de patria potestad y régimen de convivencia del menor", "familia"),

    # ADMINISTRATIVO (6)
    ("Recurso contra acto administrativo por sanción y debido proceso", "administrativo"),
    ("Derecho de petición y tutela por falta de respuesta de entidad pública", "administrativo"),
    ("Demanda de nulidad y restablecimiento del derecho contra resolución administrativa", "administrativo"),
    ("Reclamación por liquidación de impuesto y cobro coactivo", "administrativo"),
    ("Solicitud de revocatoria directa de un acto administrativo desfavorable", "administrativo"),
    ("Acción de cumplimiento por incumplimiento de norma por parte de una entidad", "administrativo"),

    # CONSUMO (6)
    ("Reclamación por producto defectuoso y garantía legal", "consumo"),
    ("Queja por publicidad engañosa y devolución de dinero", "consumo"),
    ("Reclamo por cobro no autorizado en tarjeta y reversión de pago", "consumo"),
    ("Solicitud de retracto por compra realizada por internet", "consumo"),
    ("Reclamación por cláusulas abusivas en contrato de servicio", "consumo"),
    ("Queja por incumplimiento en entrega de producto y solicitud de reembolso", "consumo"),
]


df = pd.DataFrame(data, columns=["texto", "tema"])

# 2) Split de datos
X_train, X_test, y_train, y_test = train_test_split(
    df["texto"], df["tema"], test_size=0.3, random_state=42, stratify=df["tema"]
)

# 3) Pipeline: TF-IDF + LinearSVC
spanish_stopwords = {
    "de","la","que","el","en","y","a","los","del","se","las","por","un","para","con",
    "no","una","su","al","lo","como","más","pero","sus","le","ya","o","este","sí",
    "porque","cuando","donde","qué","quien","quienes","cual","cuales"
}

model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1,2),
        stop_words=list(spanish_stopwords)
    )),
    ("clf", LinearSVC())
])

model.fit(X_train, y_train)

# 4) Evaluación
y_pred = model.predict(X_test)
print("=== Reporte de clasificación ===")
print(classification_report(y_test, y_pred, zero_division=0))


print("=== Matriz de confusión ===")
print(confusion_matrix(y_test, y_pred))

# 5) Predicción + explicación básica (términos con mayor peso)
def explicar_prediccion(texto: str, top_n: int = 8):
    """
    Explica la decisión del LinearSVC mostrando términos con mayor contribución
    hacia la clase predicha (aprox. trazabilidad por pesos).
    """
    tfidf: TfidfVectorizer = model.named_steps["tfidf"]
    clf: LinearSVC = model.named_steps["clf"]

    X_vec = tfidf.transform([texto])
    pred = clf.predict(X_vec)[0]

    feature_names = tfidf.get_feature_names_out()
    # Para LinearSVC, coef_ es [n_classes, n_features]
    classes = clf.classes_
    class_index = list(classes).index(pred)
    coefs = clf.coef_[class_index]

    # contribución: peso * valor tfidf
    vec_dense = X_vec.toarray().flatten()
    contrib = coefs * vec_dense

    top_idx = contrib.argsort()[::-1][:top_n]
    top_terms = [(feature_names[i], float(contrib[i])) for i in top_idx if vec_dense[i] > 0]

    return pred, top_terms

texto_prueba = "Quiero demandar por despido injustificado y reclamar mi indemnización."
pred, explicacion = explicar_prediccion(texto_prueba)
print("\nTexto:", texto_prueba)
print("Tema sugerido:", pred)
print("Términos relevantes:", explicacion)
