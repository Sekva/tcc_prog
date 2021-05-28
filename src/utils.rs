use crate::defs::*;
use std::convert::TryInto;

// Função que toma um vetor de tamanho variavel e devolve um array de tamanho fixo contiguo na memoria
pub fn vec_arr_fixo<T>(v_in: Vec<T>) -> [T; DIM]
where
    T: Copy,
{
    let mut v = Vec::new(); // vetor onde vai ser guardado DIM elementos do vetor

    // caso tenha menos elementos que a dimensão tratata, tem algo errado
    if v_in.len() < DIM {
        panic!("Vetor muito pequeno para transformar em array");
    }

    for i in 0..DIM {
        v.push(v_in[i]); // copia os DIM elementos
    }

    // Tenta converter no array de mesmo tamanho, caso não de, algo deu errado, não tem memoria ram suficiente?
    v.try_into().unwrap_or_else(|v: Vec<T>| {
        panic!(
            "Tentou converter um array vetor de tamanho {} em um array de tamanho {}",
            v.len(),
            DIM
        )
    })
}

// Função que retorna o gradiente da função f em um ponto x
// Gradiente sendo o vetor de derivadas parciais da função avaliadas no ponto
// Usando metodo central de derivada: D(f(x)) = (f(x-h) - f(x+h)) / (2*h)
pub fn auto_grad(x: Ponto, f: impl Fn(Ponto) -> NumReal) -> Ponto {
    // Vetor onde vai ser guardado as derivadas parciais avaliadas em x
    let mut xr = Vec::new();

    // Considera a derivada parcial à respeito a i por cada vez
    for i in 0..x.len() {
        // Cria duas copias, uma para a subitração de h e outra para a soma
        let mut x1 = x.clone();
        let mut x2 = x.clone();

        // Aplica a diferença só na componente em questão
        x1[i] -= DBL_EPS;
        x2[i] += DBL_EPS;

        // Avalia a função nos dois pontos
        let y1 = f(x1.clone());
        let y2 = f(x2.clone());

        // Calcula a variação pelo método central
        xr.push((y2 - y1) / (x2[i] - x1[i])); // x2 - x1 é o mesmo que 2*h
    }

    // Retorna o vetor como um array (contiguo na memoria)
    return vec_arr_fixo(xr);
}

// Calcula o produto interno da forma padrão
pub fn produto_interno(a: &Ponto, b: &Ponto) -> NumReal {
    let mut acc = 0.0;
    for idx in 0..DIM {
        acc += a[idx] * b[idx];
    }
    return acc;
}

// Calcula o produto interno da forma padrão para vetores de qualquer tamanaho
pub fn _produto_interno_generico(a: &Vec<NumReal>, b: &Vec<NumReal>) -> NumReal {
    let mut acc = 0.0;
    for idx in 0..(b.len().min(a.len())) {
        acc += a[idx] * b[idx];
    }
    return acc;
}

// Calcula a norma na forma padrão
pub fn norma(x: &Ponto) -> NumReal {
    let mut soma = 0.0;
    for i in x.iter() {
        soma += i * i;
    }
    soma.sqrt()
}

// Verifica se um número é proximo o suficiente de 0 pra ser considerado zero
// É considerado 0 se a distancia desse número pra 0 for menor que a quantidade "infinitesimal" usada na diferenciação
pub fn prox_o_suficiente_de_zero(n: NumReal) -> bool {
    (n - 0.0).abs() < DBL_EPS
}

// Calcula o vetor normalizado (magnetude 1) de um dado vetor
// Calcula a normal escalando o vetor pelo inverso de sua magnetude
// vetor_normal = (1/magnetude(vetor)) * vetor
pub fn normalizar(vetor: &Ponto) -> Ponto {
    // Calcula a norma do vetor (ou ponto, já que compartilham a mesma estrutura fisica)
    let norma = norma(vetor);

    // Cria uma copia do vetor pra ser retornado mais tarde
    let mut novo_vetor = vetor.clone();

    // Divide-se cada componente do vetor pela norma (ou magnetude, mesma coisa nesse caso)
    for idx in 0..DIM {
        novo_vetor[idx] /= norma;
    }

    // Retorna
    novo_vetor
}

// Calcula a distancia entre dois pontos ou dois vetores
// Poderia usar o mesmo codigo da norma? sim, mas melhor deixar aberto à outras formas de mensurar distancias
// Deixando não restrito à normal euclidiana
pub fn dist(p: Ponto, q: Ponto) -> NumReal {
    let mut acc: f64 = 0.0;
    for i in 0..DIM {
        acc += (p[i] - q[i]).powi(2);
    }
    acc.sqrt()
}

// Verifica se pelo menos dois vetores de uma lista são linearmente dependentes
pub fn sao_linearmente_dependentes(pontos: &Vec<Ponto>) -> bool {
    // Pra cada vetor, calcula os normais
    let normais: Vec<Ponto> = pontos.iter().map(|ponto| normalizar(ponto)).collect();

    // Dois vetores, a e b, são linearmente dependentes se e somente se
    // puderem ser escritos como a = k*b, para algum k real.
    // Então se a e b forem normalizados, eles deve ser iguals

    // Para cada normal
    for normal_idx in 0..normais.len() {
        // Para cada normal de novo
        for normal_idx_2 in 0..normais.len() {
            // Caso as normais escolhidas sejam diferentes
            if normal_idx != normal_idx_2 {
                // Se a distancia entre os dois vetores for desconsideravel
                if prox_o_suficiente_de_zero(dist(normais[normal_idx], normais[normal_idx_2])) {
                    return true; // Então existem pelo menos dois vetores linearmente dependentes na lista
                }
            }
        }
    }

    // Foi olhada todos os pares de vetores possiveis, e nenhum par era linearmente dependente
    false
}

// Função simples que retorna o maximo entre dois numeros, só pra
// ficar mais legivel daqui pra frente
pub fn max(a: NumReal, b: NumReal) -> NumReal {
    a.max(b)
}

// Retorna o proximo ponto do espaço limitado do problema
// Melhor assim do que fazer loops pra cada dimensão, o que nem dinamico fica
pub fn prox_ponto(
    mut x: Vec<NumReal>,
    x_min: &Vec<NumReal>,
    x_max: &Vec<NumReal>,
    dim: usize,
    passo: NumReal,
) -> Vec<NumReal> {
    x[dim - 1] += passo;

    for idx in (0..dim).rev() {
        if x[idx] > x_max[idx] {
            x[idx] = x_min[idx];

            if idx > 0 {
                x[idx - 1] += passo;
            }
        }
    }

    return x;
}

// Verifica de dois vetores são iguais
pub fn _iguais(a: &Vec<NumReal>, b: &Vec<NumReal>, dim: usize) -> bool {
    for idx in 0..dim {
        if a[idx] != b[idx] {
            // Se alguma componente não é a mesma, não são
            return false;
        }
    }
    return true;
}

// Verifica de dois vetores são iguais, considerando o erro da maquina
pub fn quase_iguais(a: &Vec<NumReal>, b: &Vec<NumReal>, dim: usize) -> bool {
    for idx in 0..dim {
        if (a[idx] - b[idx]).abs() > DBL_EPS {
            // Se alguma componente não é a mesma, não são
            return false;
        }
    }
    return true;
}

// Transposição de matirz
pub fn transpor_matriz(m: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    let mut t = vec![Vec::with_capacity(m.len()); m[0].len()];
    for r in m {
        for i in 0..r.len() {
            t[i].push(r[i]);
        }
    }
    t
}

// Gera um vetor nulo, exceto na entrada determinada com o valor determinado
// Usado inicialmente para "selecionar" um unico valor de um vetor
pub fn fn_aux_vetor_nulo_exceto(idx: usize, dim: usize, val: NumReal) -> Vec<NumReal> {
    let mut ret = vec![0.0; dim];
    ret[idx] = val;
    ret
}

// Funções auxiliares para selecionar apenas um componente
// dos vetores tᵍ, tʰ⁺ e tʰ⁻, selecionanando o proprio valor,
// ou o simetrico
pub fn fn_aux_vtgj(idx: usize, mi: usize) -> Vec<NumReal> {
    fn_aux_vetor_nulo_exceto(idx, mi, -1.0)
}
pub fn fn_aux_vrh_p(idx: usize, me: usize) -> Vec<NumReal> {
    fn_aux_vetor_nulo_exceto(idx, me, 1.0)
}
pub fn fn_aux_vrh_n(idx: usize, me: usize) -> Vec<NumReal> {
    fn_aux_vetor_nulo_exceto(idx, me, -1.0)
}

pub fn produto_escalar(a: NumReal, b: Ponto) -> Ponto {
    let mut r = [0.0; DIM];
    for i in 0..DIM {
        r[i] = a * b[i];
    }
    r
}

pub fn soma_pontos(a: Ponto, b: Ponto) -> Ponto {
    let mut r = [0.0; DIM];
    for i in 0..DIM {
        r[i] = a[i] + b[i];
    }
    r
}

pub fn subtracao_pontos(a: Ponto, b: Ponto) -> Ponto {
    let mut r = [0.0; DIM];
    for i in 0..DIM {
        r[i] = a[i] - b[i];
    }
    r
}

pub fn line_search(x: Ponto, direcao: Ponto, f: &impl Fn(Ponto) -> NumReal) -> f64 {
    //TODO: melhorar line search
    let mut y_atual = f(x);
    let mut a_atual = 0.0;

    // de 0.0 à 0.99
    for _ in 0..100 {
        let a_novo = a_atual + 0.01;
        let x_novo = soma_pontos(x, produto_escalar(a_novo, direcao));
        let y_novo = f(x_novo);

        if y_novo < y_atual {
            y_atual = y_novo;
            a_atual = a_novo;
        }
    }

    a_atual
}
use crate::defs::OP;
fn op_direta_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>, op: OP) -> Vec<Vec<NumReal>> {
    assert_eq!(a.len(), b.len());
    let mut resultante = Vec::new();

    for i in 0..a.len() {
        assert_eq!(a[i].len(), b[i].len());

        let mut linha = Vec::new();

        for j in 0..a[i].len() {
            let val = match op {
                OP::ADD => a[i][j] + b[i][j],
                OP::SUB => a[i][j] - b[i][j],
            };

            linha.push(val);
        }
        resultante.push(linha);
    }

    resultante
}

fn _subtracao_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    op_direta_matriz(a, b, OP::SUB)
}

fn soma_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    op_direta_matriz(a, b, OP::ADD)
}

fn matriz_por_escalar(a: NumReal, mut b: Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    for i in 0..b.len() {
        for j in 0..b[i].len() {
            b[i][j] = a * b[i][j];
        }
    }

    b
}

fn produto_externo(a: Ponto, b: Ponto) -> Vec<Vec<NumReal>> {
    let mut resultante = Vec::new();

    for i in 0..a.len() {
        let mut linha = Vec::new();

        for j in 0..b.len() {
            linha.push(a[i] * b[j]);
        }

        resultante.push(linha);
    }
    resultante
}

pub fn prod_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    let mut resultante = vec![vec![0.0; b[0].len()]; a.len()];

    for i in 0..resultante.len() {
        for j in 0..resultante[i].len() {
            let mut soma = 0.0;
            for k in 0..b.len() {
                soma += a[i][k] * b[k][j]
            }
            resultante[i][j] = soma;
        }
    }

    resultante
}

fn produto_matriz_vetor(matriz: &Vec<Vec<NumReal>>, vetor: &Ponto) -> Ponto {
    let mut vetor_coluna = Vec::new();

    for i in 0..DIM {
        vetor_coluna.push(vec![vetor[i]]);
    }

    let resultante = prod_matriz(matriz, &vetor_coluna);
    let vetor_plano: Vec<NumReal> = resultante.iter().map(|e| e[0]).collect();
    vec_arr_fixo(vetor_plano)
}

pub fn bfgs(hessiana: Vec<Vec<NumReal>>, s: Ponto, y: Ponto) -> Vec<Vec<NumReal>> {
    //TODO:

    // calculo da inversa
    /*
        let mut identidade = vec![vec![0.0; DIM]; DIM];
        for i in 0..DIM {
            identidade[i][i] = 1.0;
        }

        let rho: NumReal = 1.0 / produto_interno(&s, &y);

        // main BFGS update to the Hessian
        // dmat A1 = I - outer_prod(sk, yk) * rhok;
        // dmat A2 = I - outer_prod(yk, sk) * rhok;
        // dmat Hk_A2 = prod(Hk, A2);
        // dmat new_Hk = prod(A1, Hk_A2) + rhok * outer_prod(sk, sk);
        // Hk = new_Hk;

        // https://gist.github.com/rmcgibbo/4735287#file-bfgs_only_fprime-cpp-L198
        // https://math.stackexchange.com/questions/2271887/how-to-solve-the-matrix-minimization-for-bfgs-update-in-quasi-newton-optimizatio

        let a1 = subtracao_matriz(&identidade, &matriz_por_escalar(rho, produto_externo(s, y)));
        let a2 = subtracao_matriz(&identidade, &matriz_por_escalar(rho, produto_externo(y, s)));
        let b_a2 = prod_matriz(&hessiana, &a2);
        let a1_b_a2 = prod_matriz(&a1, &b_a2);
        let r_s_s = matriz_por_escalar(rho, produto_externo(s, s));
        let inversa = soma_matriz(&a1_b_a2, &r_s_s);
    */

    // calculo da normal

    // bk1 = bk + alpha*uut + beta*vvt
    // u = y
    // v = bks

    let alpha = 1.0 / produto_interno(&y, &s);
    let b_s = produto_matriz_vetor(&hessiana, &s);
    let beta = -1.0 / produto_interno(&s, &b_s);

    let alpha_uut = matriz_por_escalar(alpha, produto_externo(y, y));
    let beta_vvt = matriz_por_escalar(beta, produto_externo(b_s, b_s));
    let hessiana_atualizada = soma_matriz(&hessiana, &soma_matriz(&alpha_uut, &beta_vvt));

    hessiana_atualizada
}
