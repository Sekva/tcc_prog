use crate::defs::OP;
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

// Calcula o produto por escalar de um numero de um ponto
pub fn produto_escalar(a: NumReal, b: Ponto) -> Ponto {
    let mut r = [0.0; DIM];

    // Multiplica cada entrada do ponto pelo escalar
    for i in 0..DIM {
        r[i] = a * b[i];
    }

    r
}

// Calcula a soma de dois pontos
pub fn soma_pontos(a: Ponto, b: Ponto) -> Ponto {
    let mut r = [0.0; DIM];

    // Soma os mesmos componentes dos dois pontos e salva no ponto resultante
    for i in 0..DIM {
        r[i] = a[i] + b[i];
    }
    r
}

// Calcula a soma de dois pontos
pub fn subtracao_pontos(a: Ponto, b: Ponto) -> Ponto {
    let mut r = [0.0; DIM];
    // Subtrai os mesmos componentes dos dois pontos e salva no ponto resultante
    for i in 0..DIM {
        r[i] = a[i] - b[i];
    }
    r
}

// Função de line search para uma função
// Busca o valor otimo de ɑ entre [0, 1] de forma que
// minimize f(x + ɑ*d), d sendo a direção de busca
pub fn line_search(x: Ponto, direcao: Ponto, f: &impl Fn(Ponto) -> NumReal) -> f64 {
    //TODO: melhorar line search

    // Calcula o valor de f(x + 0*d)
    let mut y_atual = f(x);
    let mut a_atual = 0.0;
    let mut melhor_a = 0.0;

    while a_atual < 1.0 {
        // Incrementa ɑ
        a_atual = a_atual + 0.01;

        // Calcula x = x + ɑ*d
        let x_novo = soma_pontos(x, produto_escalar(a_atual, direcao));

        // Calcula f(x), que é f(x + ɑ*d)
        let y_novo = f(x_novo);

        // Se o valor for menor, esse ɑ é melhor
        if y_novo < y_atual {
            y_atual = y_novo;
            melhor_a = a_atual;
        }
    }

    melhor_a
}

// Calcula uma operação indice a indice de uma matriz
fn op_direta_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>, op: OP) -> Vec<Vec<NumReal>> {
    // Tem que ter o mesmo numero de linhas
    assert_eq!(a.len(), b.len());

    let mut resultante = Vec::new();

    // Para cada linha i
    for i in 0..a.len() {
        // Cada linha em a e b tem que ter o mesmo numero de colunas
        assert_eq!(a[i].len(), b[i].len());

        // Linha a ser adicionada
        let mut linha = Vec::new();

        // Para cada coluna da linha i de a e b
        for j in 0..a[i].len() {
            // Se for soma, soma, se não, subtrai
            let val = match op {
                OP::ADD => a[i][j] + b[i][j],
                OP::SUB => a[i][j] - b[i][j],
            };

            // Adiciona a coluna operada
            linha.push(val);
        }

        // Adiciona a linha na matriz resultante
        resultante.push(linha);
    }

    resultante
}

// Faz uma operação indice a indice usando a operação de subtração
fn _subtracao_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    op_direta_matriz(a, b, OP::SUB)
}

// Faz uma operação indice a indice usando a operação de adição
fn soma_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    op_direta_matriz(a, b, OP::ADD)
}

// Multiplica todos os indices de uma matriz por um numero
fn matriz_por_escalar(a: NumReal, mut b: Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    // Para cada linha i
    for i in 0..b.len() {
        // Para cada coluna j da linha i
        for j in 0..b[i].len() {
            // Multiplica a entrada pelo escalar
            b[i][j] = a * b[i][j];
        }
    }

    b
}

// Calcula o produto externo entre dois vetores
// (abᵀ)ᵢⱼ = aᵢ * bⱼ
fn produto_externo(a: Ponto, b: Ponto) -> Vec<Vec<NumReal>> {
    let mut resultante = Vec::new();

    //Para cada item i de a
    for i in 0..a.len() {
        // Linha da matriz resultante
        let mut linha = Vec::new();

        // Para cada item j de b
        for j in 0..b.len() {
            // A entrada ij da matriz vai ser aᵢ * bⱼ
            linha.push(a[i] * b[j]);
        }

        // Adiciona a linha na matriz resultante
        resultante.push(linha);
    }
    resultante
}

// Calcula a matriz produto de duas outras matrizes
pub fn prod_matriz(a: &Vec<Vec<NumReal>>, b: &Vec<Vec<NumReal>>) -> Vec<Vec<NumReal>> {
    // O produto tem o numero de linhas de a e o numero de colunas de b
    let mut resultante = vec![vec![0.0; b[0].len()]; a.len()];

    // Faz o calculo padrão, (AB)ᵢⱼ = soma de  aᵢₖ * bₖⱼ, com k=1, ...(numero de linhas de b)
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

// Aplica uma matriz em um ponto, Ab, A matriz, b ponto
pub fn produto_matriz_vetor(matriz: &Vec<Vec<NumReal>>, vetor: &Ponto) -> Ponto {
    let mut vetor_coluna = Vec::new();

    // Transforma o ponto em uma matriz coluna
    for i in 0..DIM {
        vetor_coluna.push(vec![vetor[i]]);
    }

    // Calcula um produto de matriz normal
    let resultante = prod_matriz(matriz, &vetor_coluna);

    // O produto é uma matriz coluna, então transforma de novo em um ponto
    let vetor_plano: Vec<NumReal> = resultante.iter().map(|e| e[0]).collect();
    vec_arr_fixo(vetor_plano)
}

// Calcula a hessiana do proximo ponto usando as informações de atualização
// Usando o método BFGS, se usa o grandiete do proximo ponto e do ponto atual
// junto com a informação de atualização do ponto (x_k+1 = x_k + a*d) onde a*d
// é a informação.
// Logo
// y = ∇f(x_k+1) - ∇f(x_k)
// s = (x_k+1) - (x_k)
// Toma como entrada a hessiana em x_k, H(x_k), s e y
// Retornando a hessiana em x_k+1, H(x_k+1)
pub fn bfgs(hessiana: Vec<Vec<NumReal>>, s: Ponto, y: Ponto) -> Vec<Vec<NumReal>> {
    /*
       // Calculo da inversa da aproximação da hessiana (não usado), considerando que a entrada também é a inversa
       // B_K+1 = B_k-1
       // https://gist.github.com/rmcgibbo/4735287#file-bfgs_only_fprime-cpp-L198
       // {\displaystyle B_{att}^{-1}=\left(I-{\frac {\mathbf {s} \mathbf {y} ^{T}}{\mathbf {y} ^{T}\mathbf {s} }}\right)B^{-1}\left(I-{\frac {\mathbf {y}\mathbf {s}^{T}}{\mathbf {y} ^{T}\mathbf {s}}}\right)+{\frac {\mathbf {s}\mathbf {s}^{T}}{\mathbf {y} ^{T}\mathbf {s}}}.}

       // Matriz Identidade
       let mut identidade = vec![vec![0.0; DIM]; DIM];
       for i in 0..DIM {
           identidade[i][i] = 1.0;
       }

       // rho é o termo comum yᵀs
       let rho: NumReal = 1.0 / produto_interno(&s, &y);

       // Calcula a matriz da esquerda de B⁻¹
       let a1 = _subtracao_matriz(&identidade, &matriz_por_escalar(rho, produto_externo(s, y)));

       // Calcula a matriz da direita de B⁻¹
       let a2 = _subtracao_matriz(&identidade, &matriz_por_escalar(rho, produto_externo(y, s)));

       // Calcula a1(Ba2)
       let b_a2 = prod_matriz(&hessiana, &a2);
       let a1_b_a2 = prod_matriz(&a1, &b_a2);

       // Calcula o termo somado
       let r_s_s = matriz_por_escalar(rho, produto_externo(s, s));

       // Por fim, soma
       let inversa = soma_matriz(&a1_b_a2, &r_s_s);
    */

    // Calculo da aproximação da hessiana com o passo de atualização:
    // B = B + alpha*uuᵀ + beta*vvᵀ
    // u = y
    // v = (Bs)
    // alpha = 1/yᵀs
    // beta = -1/(sᵀ(Bs))
    // abᵀ é o produto externo (outer product) dos vetores a e b
    // aᵀb é o produto interno euclidiano dos vetores a e b

    // Calcula Bs, visto que é usado mais de uma vez
    let b_s = produto_matriz_vetor(&hessiana, &s);

    // Calcula alpha e beta com o produto interno
    let alpha = 1.0 / produto_interno(&y, &s);
    let beta = -1.0 / produto_interno(&s, &b_s);

    // Calcula os produtos externos (que resultam em matrizes) e multiplicam pelos seus escalares
    let alpha_uut = matriz_por_escalar(alpha, produto_externo(y, y));
    let beta_vvt = matriz_por_escalar(beta, produto_externo(b_s, b_s));

    // Soma as 3 matrizes
    let hessiana_atualizada = soma_matriz(&hessiana, &soma_matriz(&alpha_uut, &beta_vvt));

    hessiana_atualizada
}
