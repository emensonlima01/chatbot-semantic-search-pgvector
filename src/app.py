from flask import Flask, request, jsonify
import psycopg2
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from datetime import date
from collections import defaultdict
from unidecode import unidecode

load_dotenv()

# --- Configuração do Modelo de Embedding ---
try:
    print("Carregando modelo de embedding...")
    embedding_model = SentenceTransformer('neuralmind/bert-base-portuguese-cased')
    print("Modelo carregado com sucesso.")
except Exception as e:
    print(f"Erro ao carregar o modelo de embedding: {e}")
    exit()

def get_embedding(text: str) -> list[float]:
    processed_text = text.lower().strip()
    if embedding_model:
        embedding = embedding_model.encode(processed_text)
        if hasattr(embedding, 'tolist'): embedding = embedding.tolist()
        if not isinstance(embedding, list): embedding = list(embedding)
        if len(embedding) != 768: raise ValueError(f"Dimensão inesperada: {len(embedding)}")
        if not all(isinstance(x, (float, np.floating)) for x in embedding): raise ValueError("Valores não-float")
        return embedding
    raise ValueError("Modelo não carregado.")

app = Flask(__name__)

# --- Configuração da Conexão com o Banco de Dados ---
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

def get_db_connection():
    conn = psycopg2.connect(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASSWORD)
    from pgvector.psycopg2 import register_vector
    register_vector(conn)
    return conn

def remover_acentos(text: str) -> str: # Função renomeada para PT-BR
    """Remove acentos de uma string."""
    if text is None:
        return ""
    return unidecode(text)

# --- Carregamento de Palavras-Chave de Intenção ---
CACHE_PALAVRAS_CHAVE_INTENCAO = defaultdict(lambda: defaultdict(list)) # Cache renomeado

def carregar_palavras_chave_intencao(): # Função renomeada
    global CACHE_PALAVRAS_CHAVE_INTENCAO
    CACHE_PALAVRAS_CHAVE_INTENCAO = defaultdict(lambda: defaultdict(list))
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Query na tabela traduzida
        cur.execute("""
            SELECT codigo_intencao, tipo_palavra_chave, valor_palavra_chave
            FROM palavras_chave_intencao
            WHERE ativo = TRUE
            ORDER BY codigo_intencao, tipo_palavra_chave, prioridade DESC, length(valor_palavra_chave) DESC;
        """)

        count = 0
        for row in cur.fetchall():
            codigo_intencao, tipo_kw, valor_kw = row # Colunas traduzidas
            normalized_kw = remover_acentos(valor_kw.lower())
            CACHE_PALAVRAS_CHAVE_INTENCAO[codigo_intencao][tipo_kw].append(normalized_kw)
            count +=1

        for codigo_intencao_val in CACHE_PALAVRAS_CHAVE_INTENCAO:
            if 'prefixo' in CACHE_PALAVRAS_CHAVE_INTENCAO[codigo_intencao_val]:
                 CACHE_PALAVRAS_CHAVE_INTENCAO[codigo_intencao_val]['prefixo'].sort(key=len, reverse=True)
            if 'separador' in CACHE_PALAVRAS_CHAVE_INTENCAO[codigo_intencao_val]:
                 CACHE_PALAVRAS_CHAVE_INTENCAO[codigo_intencao_val]['separador'].sort(key=len, reverse=True)

        print(f"Carregadas {count} palavras-chave de intenção normalizadas no cache.")
        cur.close()
    except psycopg2.Error as e:
        print(f"Erro ao carregar palavras-chave do banco: {e}")
    except Exception as e_gen:
        print(f"Erro geral ao carregar palavras-chave: {e_gen}")
    finally:
        if conn:
            conn.close()

with app.app_context():
    carregar_palavras_chave_intencao() # Chama função renomeada

def formatar_resposta_mensagem(message): # Função renomeada
    return message.replace('\n', ' ').strip()

def formatar_validade(obj_data_validade): # Função renomeada
    if isinstance(obj_data_validade, date):
        return obj_data_validade.strftime('%d/%m/%Y')
    return "N/A"

# --- Funções Auxiliares para Hierarquia de Seções (Nomes em PT-BR) ---
def obter_secao_por_nome(cur, nome_secao_normalizado): # Função e param renomeados
    cur.execute(
        "SELECT id, nome, secao_pai_id FROM secoes_catalogo WHERE unaccent(lower(nome)) = %s;",
        (nome_secao_normalizado.strip(),)
    )
    return cur.fetchone()

def obter_subsecoes_diretas(cur, id_secao_pai): # Função e param renomeados
    cur.execute(
        "SELECT id, nome FROM secoes_catalogo WHERE secao_pai_id = %s ORDER BY nome;",
        (id_secao_pai,)
    )
    return cur.fetchall()

def obter_ids_secao_e_subsecoes(cur, id_secao): # Função e param renomeados
    query = """
        WITH RECURSIVE sub_secoes AS (
            SELECT id FROM secoes_catalogo WHERE id = %s
            UNION ALL
            SELECT cs.id FROM secoes_catalogo cs INNER JOIN sub_secoes ss ON cs.secao_pai_id = ss.id
        ) SELECT id FROM sub_secoes;""" # Coluna secao_pai_id
    cur.execute(query, (id_secao,))
    return [row[0] for row in cur.fetchall()]


@app.route("/api/prompt", methods=["POST"])
def handle_prompt():
    dados_req = request.get_json() # Var renomeada
    if not dados_req or "prompt" not in dados_req:
        return jsonify({"message": "Corpo da requisição JSON deve conter um campo 'prompt'."}), 400

    texto_prompt = dados_req["prompt"].strip() # Var renomeada
    if not texto_prompt:
        return jsonify({"message": "O campo 'prompt' não pode estar vazio."}), 400

    mensagem_resposta = "" # Var renomeada
    conn = None

    try:
        vetor_embedding_prompt = get_embedding(texto_prompt)
        np_embedding_prompt = np.array(vetor_embedding_prompt)
    except Exception as e:
        print(f"Erro ao gerar embedding: {e}")
        return jsonify({"message": formatar_resposta_mensagem(f"Erro ao processar texto: {e}")}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        print(f"\nDEBUG: Prompt Original: '{texto_prompt}'")

        prompt_minusc_norm = remover_acentos(texto_prompt.lower()) # Var renomeada
        print(f"DEBUG: Prompt Normalizado: '{prompt_minusc_norm}'")

        LIMITE_ITENS_LISTAGEM = 10

        # INTENÇÃO 1: Catálogo Geral
        kws_catalogo_geral = CACHE_PALAVRAS_CHAVE_INTENCAO.get("CATALOGO_GERAL", {}).get("contem", [])
        kws_exclusao_int1 = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_ITENS_CATEGORIA", {}).get("prefixo", [])
        if any(kw in prompt_minusc_norm for kw in kws_catalogo_geral) and \
           not any(ex_kw in prompt_minusc_norm for ex_kw in kws_exclusao_int1):
            print("DEBUG: INTENÇÃO 1 DETECTADA - Catálogo Geral")
            cur.execute("SELECT nome, id FROM secoes_catalogo WHERE secao_pai_id IS NULL ORDER BY nome;")
            secoes_principais = cur.fetchall()
            if secoes_principais:
                lista_str_secoes = []
                for nome_sec, id_sec in secoes_principais:
                    filhas = obter_subsecoes_diretas(cur, id_sec)
                    if filhas: lista_str_secoes.append(f"{nome_sec} (ex: {filhas[0][1]})")
                    else: lista_str_secoes.append(nome_sec)
                mensagem_resposta = (f"Nosso catálogo principal inclui: {', '.join(lista_str_secoes)}. "
                                    "Pergunte 'o que tem em [nome da seção]?' para detalhes.")
            else: mensagem_resposta = "Catálogo de seções ainda não definido."
            cur.close(); conn.close(); conn = None
            return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200

        # INTENÇÃO 2: Listar ITENS por TIPO e MARCA
        kws_int2_prefixos = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_ITENS_TIPO_MARCA", {}).get("prefixo", [])
        kws_int2_separadores = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_ITENS_TIPO_MARCA", {}).get("separador", [])
        tipo_prod_int2_norm, marca_int2_norm = None, None # Vars normalizadas
        temp_prompt_int2_norm = prompt_minusc_norm

        for prefixo_norm in kws_int2_prefixos:
            if temp_prompt_int2_norm.startswith(prefixo_norm + " "):
                temp_prompt_int2_norm = temp_prompt_int2_norm[len(prefixo_norm)+1:].strip(); break
        for sep_norm in kws_int2_separadores:
            if sep_norm in temp_prompt_int2_norm:
                partes = temp_prompt_int2_norm.split(sep_norm, 1)
                if len(partes) == 2 and partes[0].strip() and partes[1].strip():
                    tipo_prod_int2_norm = partes[0].strip()
                    marca_int2_norm = partes[1].strip().replace("?","")
                    if marca_int2_norm.startswith(remover_acentos("marca ")):
                        marca_int2_norm = marca_int2_norm[len(remover_acentos("marca ")) :].strip()
                    break

        if tipo_prod_int2_norm and marca_int2_norm:
            print(f"DEBUG: INTENÇÃO 2 DETECTADA - Tipo (norm): '{tipo_prod_int2_norm}', Marca (norm): '{marca_int2_norm}'")
            like_tipo_unaccented = f"%{tipo_prod_int2_norm.split()[0]}%"
            like_marca_unaccented = f"%{marca_int2_norm}%"
            cur.execute("""
                SELECT i.nome, i.descricao, m.nome as nome_marca, i.preco, i.validade
                FROM itens_secao i
                JOIN marcas m ON i.marca_id = m.id
                WHERE (unaccent(lower(i.nome)) LIKE unaccent(%s) OR unaccent(lower(i.descricao)) LIKE unaccent(%s))
                  AND unaccent(lower(m.nome)) LIKE unaccent(%s)
                ORDER BY i.nome LIMIT %s;
            """, (like_tipo_unaccented, like_tipo_unaccented, like_marca_unaccented, LIMITE_ITENS_LISTAGEM))
            # ... (resto da lógica da Intenção 2, adaptando nomes de colunas se necessário ao buscar resultados)
            itens_encontrados = cur.fetchall()
            if itens_encontrados:
                # (n, d, nome_marca, p, data_validade)
                lista_itens_formatada = [f"- {item[0]} (Marca: {item[2].capitalize()}) Preço: R${item[3]:.2f} Validade: {formatar_validade(item[4])} Desc: {item[1][:50]}..." for item in itens_encontrados]
                mensagem_resposta = f"Para '{tipo_prod_int2_norm}' da marca '{marca_int2_norm.capitalize()}':\n" + "\n".join(lista_itens_formatada)
            else: mensagem_resposta = f"Não encontrei '{tipo_prod_int2_norm}' da marca '{marca_int2_norm.capitalize()}'."
            cur.close(); conn.close(); conn = None
            return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200

        # INTENÇÃO 3: Listar ITENS ou SUBCATEGORIAS por NOME DE SEÇÃO/SUBCATEGORIA
        kws_int3_prefixos = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_ITENS_CATEGORIA", {}).get("prefixo", [])
        nome_secao_extraido_int3_norm = None

        for prefixo_norm in kws_int3_prefixos:
            if prompt_minusc_norm.startswith(prefixo_norm + " "):
                texto_apos_prefixo_norm = prompt_minusc_norm[len(prefixo_norm)+1:].strip().replace("?","").strip()
                if prefixo_norm in [remover_acentos("mostre"), remover_acentos("liste")]:
                    if texto_apos_prefixo_norm.startswith(remover_acentos("itens da ")): nome_secao_extraido_int3_norm = texto_apos_prefixo_norm[len(remover_acentos("itens da ")):].strip()
                    elif texto_apos_prefixo_norm.startswith(remover_acentos("a seção ")): nome_secao_extraido_int3_norm = texto_apos_prefixo_norm[len(remover_acentos("a seção ")):].strip()
                    else: nome_secao_extraido_int3_norm = texto_apos_prefixo_norm
                else: nome_secao_extraido_int3_norm = texto_apos_prefixo_norm
                if nome_secao_extraido_int3_norm:
                    artigos_norm = [remover_acentos(art) for art in ["a ", "o ", "as ", "os "]]; [nome_secao_extraido_int3_norm := nome_secao_extraido_int3_norm[len(art_norm):].strip() for art_norm in artigos_norm if nome_secao_extraido_int3_norm.startswith(art_norm)]
                    if nome_secao_extraido_int3_norm: break
                else: nome_secao_extraido_int3_norm = None
        if not nome_secao_extraido_int3_norm and 1 <= len(prompt_minusc_norm.split()) <= 3:
            nome_secao_extraido_int3_norm = prompt_minusc_norm

        if nome_secao_extraido_int3_norm:
            print(f"DEBUG: INTENÇÃO 3 - Potencial seção (norm): '{nome_secao_extraido_int3_norm}'")
            info_secao = obter_secao_por_nome(cur, nome_secao_extraido_int3_norm)
            if info_secao:
                id_secao_alvo, nome_secao_alvo, _ = info_secao
                print(f"DEBUG: Seção '{nome_secao_alvo}' (ID: {id_secao_alvo}) encontrada.")
                subcategorias_diretas = obter_subsecoes_diretas(cur, id_secao_alvo)
                cur.execute("SELECT i.nome, m.nome, i.preco, i.validade, i.descricao FROM itens_secao i LEFT JOIN marcas m ON i.marca_id = m.id WHERE i.secao_id = %s ORDER BY i.nome LIMIT %s;", (id_secao_alvo, LIMITE_ITENS_LISTAGEM))
                itens_diretos_secao = cur.fetchall()
                resposta_formatada_itens = [f"- {item[0]} (Marca: {item[1].capitalize() if item[1] else 'N/A'}) Preço: R${item[2]:.2f} Validade: {formatar_validade(item[3])}" for item in itens_diretos_secao] if itens_diretos_secao else []
                if subcategorias_diretas:
                    nomes_subs = [s[1].capitalize() for s in subcategorias_diretas]
                    resposta_subs_str = f"A seção '{nome_secao_alvo.capitalize()}' inclui: {', '.join(nomes_subs)}. "
                    if not itens_diretos_secao: mensagem_resposta = resposta_subs_str + "Explorar qual?"
                    else: mensagem_resposta = resposta_subs_str + f"Itens diretos:\n" + "\n".join(resposta_formatada_itens)
                elif itens_diretos_secao: mensagem_resposta = f"Em '{nome_secao_alvo.capitalize()}':\n" + "\n".join(resposta_formatada_itens)
                else: mensagem_resposta = f"Seção '{nome_secao_alvo.capitalize()}' sem subcategorias ou itens."
                cur.close(); conn.close(); conn = None
                return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200
            else: nome_secao_extraido_int3_norm = None

        # INTENÇÃO 4: Listar ITENS por MARCA
        kws_int4_prefixos = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_ITENS_MARCA", {}).get("prefixo", [])
        marca_extraida_int4_norm = None
        for prefixo_norm in kws_int4_prefixos:
            if prompt_minusc_norm.startswith(prefixo_norm + " "):
                marca_extraida_int4_norm = prompt_minusc_norm[len(prefixo_norm)+1:].strip().replace("?","").strip()
                if marca_extraida_int4_norm: break
        if marca_extraida_int4_norm:
            print(f"DEBUG: INTENÇÃO 4 DETECTADA - Marca (norm): '{marca_extraida_int4_norm}'")
            like_marca_unaccented = f"%{marca_extraida_int4_norm}%"
            cur.execute("""
                SELECT i.nome, sc.nome, i.preco, i.validade
                FROM itens_secao i JOIN marcas m ON i.marca_id = m.id
                LEFT JOIN secoes_catalogo sc ON i.secao_id = sc.id
                WHERE unaccent(lower(m.nome)) LIKE unaccent(%s)
                ORDER BY sc.nome, i.nome LIMIT %s;
            """, (like_marca_unaccented, LIMITE_ITENS_LISTAGEM))
            itens_da_marca = cur.fetchall()
            if itens_da_marca:
                lista_itens_formatada = [f"- {item[0]} (Cat: {item[1].capitalize() if item[1] else 'N/A'}) Preço: R${item[2]:.2f} Validade: {formatar_validade(item[3])}" for item in itens_da_marca]
                mensagem_resposta = f"Da marca '{marca_extraida_int4_norm.capitalize()}':\n" + "\n".join(lista_itens_formatada)
            else: mensagem_resposta = f"Não encontrei itens da marca '{marca_extraida_int4_norm.capitalize()}'."
            cur.close(); conn.close(); conn = None
            return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200

        # INTENÇÃO 5: Listar ITENS por TIPO DE PRODUTO (genérico)
        kws_int5_prefixos = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_ITENS_TIPO_GENERICO", {}).get("prefixo", [])
        tipo_prod_extraido_int5_norm = None
        for prefixo_norm in kws_int5_prefixos:
            if prompt_minusc_norm.startswith(prefixo_norm + " "):
                temp_tipo_norm = prompt_minusc_norm[len(prefixo_norm)+1:].strip().replace("?","").strip()
                if prefixo_norm == remover_acentos("tem"):
                    if len(prompt_minusc_norm.split()) > 1 and temp_tipo_norm: tipo_prod_extraido_int5_norm = temp_tipo_norm; break
                elif temp_tipo_norm: tipo_prod_extraido_int5_norm = temp_tipo_norm; break
        if not tipo_prod_extraido_int5_norm and 1 <= len(prompt_minusc_norm.split()) <= 3:
            tipo_prod_extraido_int5_norm = prompt_minusc_norm

        if tipo_prod_extraido_int5_norm:
            info_secao_check = obter_secao_por_nome(cur, tipo_prod_extraido_int5_norm)
            if info_secao_check: print(f"DEBUG: INTENÇÃO 5 - Tipo '{tipo_prod_extraido_int5_norm}' é seção. Pulando.")
            else:
                print(f"DEBUG: INTENÇÃO 5 DETECTADA - Tipo genérico (norm): '{tipo_prod_extraido_int5_norm}'")
                # ... (lógica da Intenção 5, adaptada) ...
                like_tipo_unaccented = f"%{tipo_prod_extraido_int5_norm.split()[0]}%"
                singular_base = tipo_prod_extraido_int5_norm
                if singular_base.endswith('s') and not singular_base.endswith('is'): singular_base = singular_base[:-1]
                like_singular_unaccented = f"%{singular_base}%"
                cur.execute("""
                    SELECT i.nome, m.nome, sc.nome, i.preco, i.validade
                    FROM itens_secao i LEFT JOIN marcas m ON i.marca_id = m.id
                    LEFT JOIN secoes_catalogo sc ON i.secao_id = sc.id
                    WHERE (unaccent(lower(i.nome)) LIKE unaccent(%s) OR unaccent(lower(i.nome)) LIKE unaccent(%s) OR
                           unaccent(lower(i.descricao)) LIKE unaccent(%s) OR unaccent(lower(i.descricao)) LIKE unaccent(%s))
                    ORDER BY RANDOM() LIMIT %s;
                """, (like_tipo_unaccented, like_singular_unaccented, like_tipo_unaccented, like_singular_unaccented, LIMITE_ITENS_LISTAGEM))
                itens_por_tipo = cur.fetchall()
                if itens_por_tipo:
                    lista_itens_formatada = [f"- {item[0]} (Marca: {item[1].capitalize() if item[1] else 'N/A'}, Cat: {item[2].capitalize() if item[2] else 'N/A'}) Preço: R${item[3]:.2f} Validade: {formatar_validade(item[4])}" for item in itens_por_tipo]
                    mensagem_resposta = f"Sobre '{tipo_prod_extraido_int5_norm}', encontrei:\n" + "\n".join(lista_itens_formatada)
                else: mensagem_resposta = f"Não encontrei itens do tipo '{tipo_prod_extraido_int5_norm}'."
                cur.close(); conn.close(); conn = None
                return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200

        # INTENÇÃO 6: Listar MARCAS por TIPO/CATEGORIA
        kws_int6_prefixos = CACHE_PALAVRAS_CHAVE_INTENCAO.get("LISTAR_MARCAS_POR_TIPO", {}).get("prefixo", [])
        tipo_cat_extraido_int6_norm = None
        for kw_norm in kws_int6_prefixos:
            if prompt_minusc_norm.startswith(kw_norm + " "):
                tipo_cat_extraido_int6_norm = prompt_minusc_norm[len(kw_norm)+1:].strip().replace("?","").strip()
                if tipo_cat_extraido_int6_norm: break
        if tipo_cat_extraido_int6_norm:
            print(f"DEBUG: INTENÇÃO 6 DETECTADA - Marcas de (norm): '{tipo_cat_extraido_int6_norm}'")
            info_secao_int6 = obter_secao_por_nome(cur, tipo_cat_extraido_int6_norm)
            marcas_sql, params_sql = "", []
            if info_secao_int6:
                ids_secoes = obter_ids_secao_e_subsecoes(cur, info_secao_int6[0])
                marcas_sql = "SELECT DISTINCT m.nome FROM marcas m JOIN itens_secao i ON m.id = i.marca_id WHERE i.secao_id = ANY(%s) ORDER BY m.nome;"
                params_sql = (ids_secoes,)
            else:
                like_tipo_unaccented = f"%{tipo_cat_extraido_int6_norm.split()[0]}%"
                singular_base = tipo_cat_extraido_int6_norm
                if singular_base.endswith('s') and not singular_base.endswith('is'): singular_base = singular_base[:-1]
                like_singular_unaccented = f"%{singular_base}%"
                marcas_sql = """
                    SELECT DISTINCT m.nome FROM marcas m JOIN itens_secao i ON m.id = i.marca_id
                    WHERE (unaccent(lower(i.nome)) LIKE unaccent(%s) OR unaccent(lower(i.nome)) LIKE unaccent(%s))
                    ORDER BY m.nome;"""
                params_sql = (like_tipo_unaccented, like_singular_unaccented)
            cur.execute(marcas_sql, params_sql)
            marcas_encontradas = [row[0] for row in cur.fetchall()]
            contexto_nome = info_secao_int6[1].capitalize() if info_secao_int6 else tipo_cat_extraido_int6_norm
            if marcas_encontradas: mensagem_resposta = f"Marcas para '{contexto_nome}': {', '.join(marcas_encontradas)}."
            else: mensagem_resposta = f"Não encontrei marcas para '{contexto_nome}'."
            cur.close(); conn.close(); conn = None
            return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200

        # INTENÇÃO 7: Fallback (Embedding)
        print("DEBUG: Nenhuma intenção específica (1-6) atendida. Fallback (Intenção 7).")
        if not conn: conn = get_db_connection(); cur = conn.cursor()
        # Query adaptada para nomes PT-BR
        cur.execute("SELECT i.nome, i.descricao, i.embedding <-> %s AS dist, i.preco, i.validade, m.nome FROM itens_secao i LEFT JOIN marcas m ON i.marca_id = m.id ORDER BY dist LIMIT 1;", (np_embedding_prompt,))
        melhor_produto_geral = cur.fetchone()
        cur.execute("SELECT nome, id, secao_pai_id, embedding <-> %s AS dist FROM secoes_catalogo ORDER BY dist LIMIT 1;", (np_embedding_prompt,))
        melhor_categoria_geral_row = cur.fetchone()
        dist_produto = melhor_produto_geral[2] if melhor_produto_geral else float('inf')
        dist_categoria = melhor_categoria_geral_row[3] if melhor_categoria_geral_row else float('inf')
        LIMIAR_FALLBACK_PRODUTO = 7.5
        LIMIAR_FALLBACK_CATEGORIA = 8.5
        print(f"DEBUG Fallback: Dist Produto: {dist_produto}, Dist Categoria: {dist_categoria}")

        if dist_produto < LIMIAR_FALLBACK_PRODUTO and dist_produto < (dist_categoria - 0.2):
            n,d,_,p,v,m_nome = melhor_produto_geral # nome, descricao, dist, preco, validade, nome_marca
            mensagem_resposta = f"Encontrei '{n}'. Desc: {d if d else 'N/A'}. Marca: {m_nome.capitalize() if m_nome else 'N/A'}. Preço: R${p:.2f}. Validade: {formatar_validade(v)}."
        elif dist_categoria < LIMIAR_FALLBACK_CATEGORIA:
            nome_cat, id_cat, _, _ = melhor_categoria_geral_row
            subs = obter_subsecoes_diretas(cur, id_cat)
            if subs: mensagem_resposta = f"Relacionado à seção '{nome_cat.capitalize()}', que inclui: {', '.join([s[1].capitalize() for s in subs])}. Explorar?"
            else: mensagem_resposta = f"Relacionado à seção '{nome_cat.capitalize()}'. Ver itens?"
        else: mensagem_resposta = "Desculpe, não entendi bem. Poderia reformular?"
        if conn: cur.close(); conn.close(); conn = None

    except psycopg2.Error as e:
        print(f"Erro de DB: {e}"); mensagem_resposta = "Problema ao acessar catálogo."
        if conn: conn.close(); conn = None
        return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 503
    except Exception as e:
        print(f"Erro inesperado: {e}"); import traceback; traceback.print_exc()
        mensagem_resposta = "Ocorreu um erro inesperado."
        if conn: conn.close(); conn = None
        return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 500
    finally:
        if conn: print("DEBUG: Conexão aberta no finally."); cur.close(); conn.close()

    return jsonify({"message": formatar_resposta_mensagem(mensagem_resposta)}), 200

if __name__ == "__main__":
    app.run(debug=True, port=5000)
