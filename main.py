from ollama import chat, ChatResponse
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("all-MiniLM-L6-v2").to(device)

client = QdrantClient("http://localhost:6333")

historico_mensagens = []


faq_albion = [
    {
        "pergunta": "Quais são os canais de suporte do jogo?",
        "resposta": "Antes de mais nada, consulte o FAQ Original do jogo. Se ainda sim tiver problemas, tente pela página de suporte ou mande um email diretamente para support@albiononline.com."
    },
    {
        "pergunta": "Quando o Albion Online vai ser lançado?",
        "resposta": "Albion Online vai ser lançado dia 17 de Julho. As horas de lançamento para os pacotes de fundadores vão ser anunciadas em breve."
    },
    {
        "pergunta": "É outro teste beta? Vai ter mais wipe?",
        "resposta": "Não, não vai ter mais wipe depois do lançamento, agora é pra valer!"
    },
    {
        "pergunta": "O jogo vai ser free (free to play, F2P)/grátis/gratuito/de graça quando lançar?",
        "resposta": "Não, vai ser preciso comprar um dos pacotes de fundador para jogar o jogo. Mais informações aqui."
    },
    {
        "pergunta": "Todos os benefícios do meu pacote de fundador serão restaurados durante a release?",
        "resposta": "Sim, você receberá todos os itens do pacote de fundador de volta quando lançar o jogo, incluindo ouro, montarias, itens e tudo mais."
    },
    {
        "pergunta": "E quanto ao ouro que eu comprei na loja oficial do Albion, vai ser retornado?",
        "resposta": "Sim, vai ser retornado."
    },
    {
        "pergunta": "E quanto a todos os bônus de indicação?",
        "resposta": "Também, todos retornados após o release."
    },
    {
        "pergunta": "Quanto tempo demora para receber uma recompensa de indicação?",
        "resposta": "Geralmente 30 dias após o seu amigo ter adquirido o pacote para começar a jogar Albion."
    },
    {
        "pergunta": "Comprei o jogo através do pagamento X, quanto tempo demora para processar/receber o pagamento?",
        "resposta": "Depende, pagamentos feitos com Cartão de Crédito (direto) ou PayPal são processados geralmente no mesmo dia, outros meios de pagamento demoram mais. A grande demora nos meios de pagamento é feita por conta da verificação de segurança: toda compra acarreta uma verificação de segurança e às vezes essa pode demorar um pouquinho. Mais informações sobre os meios de pagamento podem ser obtidas aqui e aqui."
    }
]


def deletar_colecao(client, collection_name):
    try:
        client.delete_collection(collection_name=collection_name)
        print(f"Coleção '{collection_name}' deletada com sucesso.")
    except Exception as e:
        print(f"Erro ao deletar a coleção '{collection_name}': {e}")


def armazenar_embeddings():
    embeddings = []
    for item in faq_albion:
        text = item['pergunta'] + " " + item['resposta']
        embedding = model.encode(text, convert_to_tensor=True).to(device)
        embeddings.append(embedding)

    collection_name = "faq_albion"
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config={"size": 384, "distance": "Cosine"}
        )
        print(f"Coleção '{collection_name}' criada com sucesso.")
    except Exception as e:
        print(f"Coleção '{collection_name}' já existe. Ignorando criação. Erro: {e}")

    for i, embedding in enumerate(embeddings):
        existing_point = client.search(
            collection_name=collection_name,
            query_vector=embedding.tolist(),
            limit=1
        )

        if not existing_point:
            client.upsert(
                collection_name=collection_name,
                points=[
                    {"id": i, "vector": embedding.tolist(), "payload": faq_albion
            
    [i]}
                ]
            )

    print("Embeddings armazenados com sucesso no Qdrant!")



def buscar_documentos_relevantes(query, top_k=3, similaridade_minima=0.5):
    query_embedding = model.encode(query, convert_to_tensor=True).to(device)
    results = client.search(
        collection_name="faq_albion",
        query_vector=query_embedding.tolist(),
        limit=top_k
    )

    documentos_relevantes = [result.payload for result in results if result.score >= similaridade_minima]

    return documentos_relevantes



def consultar_modelo_local(consulta_usuario, documentos_relevantes, historico_mensagens):
    dados_faq = "\n".join([doc['resposta'] for doc in documentos_relevantes])
    full_prompt = f"Dados do FAQ:\n{dados_faq}\nPergunta do usuário: {consulta_usuario}"

    historico_mensagens.append({"role": "user", "content": consulta_usuario})

    response: ChatResponse = chat(model="llama3.2", messages=historico_mensagens + [
        {"role": "system", "content": "Você é um assistente útil que responde perguntas com base em FAQs fornecidas."},
        {"role": "user", "content": full_prompt}
    ], options={"temperature": 0.6})

    resposta_content = response.message.content

    historico_mensagens.append({"role": "assistant", "content": resposta_content})

    return resposta_content



def executar_fluxo(consulta_usuario, top_k=3, historico_mensagens=[]):
    documentos_relevantes = buscar_documentos_relevantes(consulta_usuario, top_k)
    resposta_final = consultar_modelo_local(consulta_usuario, documentos_relevantes, historico_mensagens)
    return resposta_final


def inicializar():
    armazenar_embeddings()


if __name__ == "__main__":
    inicializar()
    # deletar_colecao(client, "faq_albion")

    print("Bem-vindo ao sistema de FAQ. Faça sua pergunta ou digite 'q' para sair.")

    while True:
        pergunta = input("\nVocê: ")
        if pergunta.lower() == "q":
            print("Encerrando o sistema de FAQ. Até logo!")
            break

        resposta_final = executar_fluxo(pergunta, historico_mensagens=historico_mensagens)

        print(f"\nCTIC: {resposta_final}")