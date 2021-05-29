use crate::defs::*;
use crate::utils::*;
use std::slice::Iter;

// Registro de informações uteis durante a geração
// das matrizes e vetores
struct InformacoesProblema {
    grad_funcao_objetivo: Ponto,
    funcao_desigualdades_avaliadas: Vec<NumReal>,
    funcao_igualdades_avaliadas: Vec<NumReal>,
    grads_funcao_desigualdades: Vec<Ponto>,
    grads_funcao_igualdades: Vec<Ponto>,
    n: usize,
    mi: usize,
    me: usize,
    dl: Ponto,
    du: Ponto,
    direcoes_encontradas: Vec<Ponto>,
    hessiana_atual: Vec<Vec<NumReal>>,
}

impl InformacoesProblema {
    fn novo(
        problema: &Problema,
        x: Ponto,
        direcoes_encontradas: &Vec<Ponto>,
        hessiana_atual: &Vec<Vec<NumReal>>,
    ) -> InformacoesProblema {
        // Gera algumas informações a partir
        // das funções avaliadas no ponto atual
        let (
            _,
            grad_funcao_objetivo,
            funcao_desigualdades_avaliadas,
            funcao_igualdades_avaliadas,
            grads_funcao_desigualdades,
            grads_funcao_igualdades,
        ) = problema.avaliar_em(x);

        // Dimenções do problema
        let n = grad_funcao_objetivo.len();
        let mi = grads_funcao_desigualdades.len();
        let me = grads_funcao_igualdades.len();

        // Limites das direções do problema
        let dl = problema.d_l;
        let du = problema.d_u;

        // Direcoes encontradas durante as iterações lineares
        let direcoes_encontradas = direcoes_encontradas.clone();

        // Hessiana da iteração atual (restrição 1c)
        let hessiana_atual = hessiana_atual.clone();

        // Cria o registro das informações e retorna
        InformacoesProblema {
            grad_funcao_objetivo,
            funcao_desigualdades_avaliadas,
            funcao_igualdades_avaliadas,
            grads_funcao_desigualdades,
            grads_funcao_igualdades,
            n,
            mi,
            me,
            dl,
            du,
            direcoes_encontradas,
            hessiana_atual,
        }
    }
}

// Gera uma linha da matriz A a partir das listas dos coeficientes
// Não necessária, mas fica melhor por questões de legibilidade
fn gerar_linha_matriz(
    coeficientes_d: Iter<NumReal>,
    coeficientes_tg: Iter<NumReal>,
    coeficientes_thp: Iter<NumReal>,
    coeficientes_thm: Iter<NumReal>,
) -> Vec<NumReal> {
    let mut linha: Vec<NumReal> = Vec::new();

    linha.extend(coeficientes_d); // Coeficientes de d
    linha.extend(coeficientes_tg); // Coeficientes de tᵍ
    linha.extend(coeficientes_thp); // Coeficientes de tʰ⁺
    linha.extend(coeficientes_thm); // Coeficientes de tʰ⁻

    linha
}

fn restricao_1a(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1a
    // gⱼ(x) + ∇gⱼ(x)ᵀd ≤ tᵍⱼ
    // Variavéis de um lado e constantes de outro
    // ∇gⱼ(x)ᵀd - tᵍⱼ ≤ -gⱼ(x)
    // Então
    // Os coeficientes de d são os valores do gradiente da j-ésima função de desigualdade
    // Os coeficientes dos tᵍ são todos nulos, com exceção do j-ésimo, que é -1
    // Os coeficientes dos tʰ⁺ são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é o valor inverso da j-ésima função de
    // de desigualdade avaliada no ponto

    // Para cada linha da restrição
    for j in 0..info.mi {
        a.push(gerar_linha_matriz(
            info.grads_funcao_desigualdades[j].iter(), // Coeficientes de d
            fn_aux_vtgj(j, info.mi).iter(),            // Coeficientes de tᵍ
            vec![0.0; info.me].iter(),                 // Coeficientes de tʰ⁺
            vec![0.0; info.me].iter(),                 // Coeficientes de tʰ⁻
        ));
        b.push(-info.funcao_desigualdades_avaliadas[j]); // Valor de b
    }

    (a, b)
}

fn restricao_1b(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1b
    // hᵣ(x) + ∇hᵣ(x)ᵀd = tʰ⁺ᵣ - tʰ⁻ᵣ
    // Variavéis de um lado e constantes de outro
    // ∇hᵣ(x)ᵀd - tʰ⁺ᵣ + tʰ⁻ᵣ = -hᵣ(x)
    // Mas é necessário que seja uma desigualdade do tipo menor que
    // Então
    // ∇hᵣ(x)ᵀd - tʰ⁺ᵣ + tʰ⁻ᵣ ≤ -hᵣ(x)
    // -∇hᵣ(x)ᵀd + tʰ⁺ᵣ - tʰ⁻ᵣ ≤ hᵣ(x)
    // Então

    // Para ∇hᵣ(x)ᵀd - tʰ⁺ᵣ + tʰ⁻ᵣ ≤ -hᵣ(x):
    // Os coeficientes de d são os valores do gradiente da r-ésima função de igualdade
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos, com exceção do r-ésimo, que é -1
    // Os coeficientes dos tʰ⁻ são todos nulos, com exceção do r-ésimo, que é 1
    // A constante da restrição é o valor inverso da r-ésima função de
    // de igualdade avaliada no ponto

    for r in 0..info.me {
        a.push(gerar_linha_matriz(
            info.grads_funcao_igualdades[r].iter(),
            vec![0.0; info.mi].iter(),
            fn_aux_vrh_n(r, info.me).iter(),
            fn_aux_vrh_p(r, info.me).iter(),
        ));

        b.push(-info.funcao_igualdades_avaliadas[r]);
    }

    // Para -∇hᵣ(x)ᵀd + tʰ⁺ᵣ - tʰ⁻ᵣ ≤ hᵣ(x):
    // Os coeficientes de d são os valores inversos do gradiente da r-ésima função de igualdade
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos, com exceção do r-ésimo, que é 1
    // Os coeficientes dos tʰ⁻ são todos nulos, com exceção do r-ésimo, que é -1
    // A constante da restrição é o valor da r-ésima função de
    // de igualdade avaliada no ponto

    for r in 0..info.me {
        let grads_invertidos: Vec<NumReal> = info.grads_funcao_igualdades[r]
            .iter()
            .map(|el| -1.0 * el)
            .collect();

        a.push(gerar_linha_matriz(
            grads_invertidos.iter(),
            vec![0.0; info.mi].iter(),
            fn_aux_vrh_p(r, info.me).iter(),
            fn_aux_vrh_n(r, info.me).iter(),
        ));

        b.push(info.funcao_igualdades_avaliadas[r]);
    }

    (a, b)
}

fn restricao_1c(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1c
    // (dᵣ)ᵀHd = 0, r =1, ..., i-1
    // Ou, por ser um produto interno em relação a H
    // (d)ᵀHdᵣ = 0, r=1, ..., i-1
    // Ou ainda
    // (Hdᵣ)ᵀd = 0, r=1, ..., i-1
    // Então
    // (Hdᵣ)ᵀd ≤ 0, r=1, ..., i-1
    // -((Hdᵣ)ᵀd) ≤ 0, r=1, ..., i-1

    // Para ambos
    // Os coeficientes de d são os valores de Hdᵣ para o primeiro caso e -(Hdᵣ) para o segundo
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante de desigualdade da restrição é 0 para ambos os casos

    // Para cada direção já encontrada
    for d_r in &info.direcoes_encontradas {
        let h_dr = produto_matriz_vetor(&info.hessiana_atual, &d_r.clone());
        let h_dr_i: Vec<NumReal> = h_dr.iter().map(|el| -1.0 * el).collect();

        a.push(gerar_linha_matriz(
            h_dr.iter(),               // Coeficientes de d
            vec![0.0; info.mi].iter(), // Coeficientes de tᵍ
            vec![0.0; info.me].iter(), // Coeficientes de tʰ⁺
            vec![0.0; info.me].iter(), // Coeficientes de tʰ⁻
        ));
        b.push(0.0); // Valor de b

        a.push(gerar_linha_matriz(
            h_dr_i.iter(),             // Coeficientes de d
            vec![0.0; info.mi].iter(), // Coeficientes de tᵍ
            vec![0.0; info.me].iter(), // Coeficientes de tʰ⁺
            vec![0.0; info.me].iter(), // Coeficientes de tʰ⁻
        ));
        b.push(0.0); // Valor de b
    }

    (a, b)
}

fn restricao_1d(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1d
    // dₗ ≤ d ≤ dᵤ
    // Que dividindo em duas
    // dₗ ≤ d
    // d ≤ dᵤ
    // Variavéis de um lado e constantes de outro
    // -d ≤ -dₗ
    // d ≤ dᵤ
    // Então

    // Para -d ≤ -dₗ:
    // Os coeficientes de d é -1 para cada i-ésima componente
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é o valor inverso da i-ésima componente de dₗ

    for i in 0..info.n {
        a.push(gerar_linha_matriz(
            fn_aux_vetor_nulo_exceto(i, info.n, -1.0).iter(),
            vec![0.0; info.mi].iter(),
            vec![0.0; info.me].iter(),
            vec![0.0; info.me].iter(),
        ));
        b.push(-info.dl[i])
    }

    // Para d ≤ dᵤ:
    // Os coeficientes de d é 1 para cada i-ésima componente
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é o valor da i-ésima componente de dᵤ

    for i in 0..info.n {
        a.push(gerar_linha_matriz(
            fn_aux_vetor_nulo_exceto(i, info.n, 1.0).iter(),
            vec![0.0; info.mi].iter(),
            vec![0.0; info.me].iter(),
            vec![0.0; info.me].iter(),
        ));
        b.push(info.du[i])
    }

    (a, b)
}

fn restricao_1e(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1e
    // 0 ≤ tᵍⱼ ≤ max(0, gⱼ(x))
    // Que dividindo em duas
    // 0 ≤ tᵍⱼ
    // tᵍⱼ ≤ max(0, gⱼ(x))
    // Variavéis de um lado e constantes de outro
    // -tᵍⱼ ≤ 0
    // tᵍⱼ ≤ max(0, gⱼ(x))
    // Então

    // Para -tᵍⱼ ≤ 0:
    // Os coeficientes de d são todos nulos
    // Os coeficientes dos tᵍ são todos nulos, com exceção do j-ésimo, que é -1
    // Os coeficientes dos tʰ⁺ são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é 0
    for j in 0..info.mi {
        a.push(gerar_linha_matriz(
            vec![0.0; info.n].iter(),
            fn_aux_vetor_nulo_exceto(j, info.mi, -1.0).iter(),
            vec![0.0; info.me].iter(),
            vec![0.0; info.me].iter(),
        ));

        b.push(0.0);
    }

    // Para tᵍⱼ ≤ max(0, gⱼ(x)):
    // Os coeficientes de d são todos nulos
    // Os coeficientes dos tᵍ são todos nulos, com exceção do j-ésimo, que é 1
    // Os coeficientes dos tʰ⁺ são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é o maior valor entre 0 e a j-ésima restriçao de desigualdade
    // avaliada no ponto
    for j in 0..info.mi {
        a.push(gerar_linha_matriz(
            vec![0.0; info.n].iter(),
            fn_aux_vetor_nulo_exceto(j, info.mi, 1.0).iter(),
            vec![0.0; info.me].iter(),
            vec![0.0; info.me].iter(),
        ));

        b.push(max(info.funcao_desigualdades_avaliadas[j], 0.0));
    }

    (a, b)
}

fn restricao_1f(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1g
    // 0 ≤ tʰ⁺ᵣ ≤ |hᵣ(x)|
    // Que dividindo em duas
    // 0 ≤ tʰ⁺ᵣ
    // tʰ⁺ᵣ ≤ |hᵣ(x)|
    // Variavéis de um lado e constantes de outro
    // -tʰ⁺ᵣ ≤ 0
    // tʰ⁺ᵣ ≤ |hᵣ(x)|
    // Então

    // Para -tʰ⁺ᵣ ≤ 0:
    // Os coeficientes de d são todos nulos
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos, com exceção do r-ésimo, que é -1
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é 0
    for r in 0..info.me {
        a.push(gerar_linha_matriz(
            vec![0.0; info.n].iter(),
            vec![0.0; info.mi].iter(),
            fn_aux_vetor_nulo_exceto(r, info.me, -1.0).iter(),
            vec![0.0; info.me].iter(),
        ));

        b.push(0.0);
    }

    // Para tʰ⁺ᵣ ≤ |hᵣ(x)|:
    // Os coeficientes de d são todos nulos
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺ são todos nulos, com exceção do r-ésimo, que é 1
    // Os coeficientes dos tʰ⁻ são todos nulos
    // A constante da restrição é o valor absoluto da r-ésima restriçao de igualdade
    // avaliada no ponto
    for r in 0..info.me {
        a.push(gerar_linha_matriz(
            vec![0.0; info.n].iter(),
            vec![0.0; info.mi].iter(),
            fn_aux_vetor_nulo_exceto(r, info.me, 1.0).iter(),
            vec![0.0; info.me].iter(),
        ));

        b.push(info.funcao_igualdades_avaliadas[r].abs());
    }

    (a, b)
}

fn restricao_1g(info: &InformacoesProblema) -> (Vec<Vec<NumReal>>, Vec<NumReal>) {
    let mut a = Vec::new();
    let mut b = Vec::new();

    // Restrição 1e
    // 0 ≤ tʰ⁻ᵣ ≤ |hᵣ(x)|
    // Que dividindo em duas
    // 0 ≤ tʰ⁻ᵣ
    // tʰ⁻ᵣ ≤ |hᵣ(x)|
    // Variavéis de um lado e constantes de outro
    // -tʰ⁻ᵣ ≤ 0
    // tʰ⁻ᵣ ≤ |hᵣ(x)|
    // Então

    // Para -tʰ⁻ᵣ ≤ 0:
    // Os coeficientes de d são todos nulos
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺  são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos, com exceção do r-ésimo, que é -1
    // A constante da restrição é 0
    for r in 0..info.me {
        a.push(gerar_linha_matriz(
            vec![0.0; info.n].iter(),
            vec![0.0; info.mi].iter(),
            vec![0.0; info.me].iter(),
            fn_aux_vetor_nulo_exceto(r, info.me, -1.0).iter(),
        ));

        b.push(0.0);
    }

    // Para tʰ⁻ᵣ ≤ |hᵣ(x)|:
    // Os coeficientes de d são todos nulos
    // Os coeficientes dos tᵍ são todos nulos
    // Os coeficientes dos tʰ⁺  são todos nulos
    // Os coeficientes dos tʰ⁻ são todos nulos, com exceção do r-ésimo, que é 1
    // A constante da restrição é o valor absoluto da r-ésima restriçao de igualdade
    // avaliada no ponto
    for r in 0..info.me {
        a.push(gerar_linha_matriz(
            vec![0.0; info.n].iter(),
            vec![0.0; info.mi].iter(),
            vec![0.0; info.me].iter(),
            fn_aux_vetor_nulo_exceto(r, info.me, 1.0).iter(),
        ));

        b.push(info.funcao_igualdades_avaliadas[r].abs());
    }

    (a, b)
}

// Gera as matrizes e vetores para um problema linear
// do tipo
// min cᵀx
// s.a.: Ax ≥ b
pub fn matriz_e_vetores_problema_linear(
    problema: &Problema,
    x: Ponto,
    lista_direcoes: &Vec<Ponto>,
    hessiana_atual: &Vec<Vec<NumReal>>,
) -> (Vec<Vec<NumReal>>, Vec<NumReal>, Vec<NumReal>) {
    // Informações uteis durante o processo de geração das informções
    let info = InformacoesProblema::novo(problema, x, lista_direcoes, hessiana_atual);

    // a é uma lista de lista de números reais, isto é, uma matriz
    // que representa A, que são os coeficientes de cada expressão
    // linear do problema linear
    let mut a: Vec<Vec<NumReal>> = Vec::new();

    // b é uma lista de números reais, um vetor, representando b,
    // que são uma lista de números reais para cada expressão linear
    // em A deve ser maior
    let mut b: Vec<NumReal> = Vec::new();

    // c é uma lista de números reais, um vetor, representando c,
    // que são os coeficientes da função objetivo do problema linear
    // Todos os componentes são a constante C, exceto pelos n primeiros elementos
    let mut c: Vec<NumReal> = vec![C; info.n + info.mi + info.me + info.me];
    for i in 0..info.n {
        // Troca a constante C por componentes do gradiente da função objetivo
        c[i] = info.grad_funcao_objetivo[i];
    }

    // Restrições a serem aplicadas, construidas de arcordo com o artigo
    let restricoes = [
        restricao_1a,
        restricao_1b,
        restricao_1c,
        restricao_1d,
        restricao_1e,
        restricao_1f,
        restricao_1g,
    ];

    // Para cada restrição, é adiquirido os blocos da matriz A e do vetor b
    // referentes a restição
    for restricao in restricoes.iter() {
        // Adiquirindo os blocos
        let (ba, bb) = restricao(&info);
        // Adicionando os blocos à matriz A e o vetor b
        let _: Vec<()> = ba.iter().map(|el| a.push(el.clone())).collect();
        let _: Vec<()> = bb.iter().map(|el| b.push(el.clone())).collect();
        // É necessário salvar a lista em branco em uma variavél por causa de lazy-evaluation
    }

    // Por segurança, verifica se o numero de linhas de A é igual ao número de elementos em b
    assert_eq!(a.len(), b.len());

    // Transforma o problema de Ax ≤ b em Ax ≥ b
    // multiplicando todos os elementos de A e b por -1
    b = b.iter().map(|el| -1.0 * el).collect();
    a = a
        .iter()
        .map(|linha| linha.iter().map(|el| -1.0 * el).collect())
        .collect();

    // Retorna a, b e c
    (a, b, c)
}
