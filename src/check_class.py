# test_estimator.py

from sklearn.utils.estimator_checks import check_estimator

# Importe a sua classe customizada que está causando o warning
from mlp import Standalone_RealMLP_TD_S_Classifier


def test_realmlp_conformance():
    """
    Este teste verifica se o nosso estimador customizado segue
    todas as regras e convenções do scikit-learn.
    """
    print("Iniciando verificação com check_estimator...")

    # Cria uma instância do seu estimador
    estimator = Standalone_RealMLP_TD_S_Classifier()

    # Roda a bateria de testes do scikit-learn
    # Se algo estiver errado, esta função vai gerar um erro detalhado.
    try:
        check_estimator(estimator)
        print("\nSUCESSO! O estimador passou em todos os testes de conformidade do scikit-learn.")
    except Exception as e:
        print(f"\nFALHA! O estimador não passou nos testes. Erro encontrado:")
        # Imprime o erro específico que o check_estimator encontrou
        raise e


if __name__ == '__main__':
    # Para rodar este teste, você talvez precise instalar o pytest:
    # pip install pytest

    test_realmlp_conformance()
