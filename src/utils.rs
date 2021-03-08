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
pub fn auto_grad(x: Ponto, f: Funcao) -> Ponto {
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

// Calcula a norma da forma padrão
fn norma(x: &Ponto) -> NumReal {
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

pub fn iguais(a: &Vec<NumReal>, b: &Vec<NumReal>, dim: usize) -> bool {
    for idx in 0..dim {
        if a[idx] != b[idx] {
            return false;
        }
    }

    return true;
}
