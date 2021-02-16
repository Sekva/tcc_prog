// Aliases de tipo, pra facilitar o entendimento
type NumReal = f64;
type Ponto = [NumReal; DIM];
type Funcao = fn(Ponto) -> NumReal; // o tipo Funcao é um ponteiro de uma função de recebe um Ponto e retorna um NumReal

use std::convert::TryInto;

// Função que toma um vetor de tamanho variavel e devolve um array de tamanho fixo contiguo na memoria
fn vec_arr_fixo<T>(v_in: Vec<T>) -> [T; DIM]
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

// Estrutura do problema
pub struct Problema {
    pub funcao_objetivo: Funcao, // A função objetivo que quer ser minimizada
    pub restricoes_igualdades: Vec<Funcao>, // A lista das m_e funções restrições de igualdades, onde h_r(x) == 0, para r = 1, ..., m_e
    pub restricoes_desigualdades: Vec<Funcao>, // A lista das m_i funções restrições de desigualdades, onde g_j(x) == 0, para j = 1, ..., m_i

    // Pontos que restrigem o espaço total do problema
    // Está aqui porque no EMFCQ, procurar no R^n todo é impossivel
    // E a implementação da verificação do EMFCQ exige que existam limites fixados
    pub x_min: Ponto,
    pub x_max: Ponto,
}

impl Problema {
    // Retorna o proximo ponto do espaço limitado do problema
    // Melhor assim do que fazer loops pra cada dimensão, o que nem dinamico fica
    pub fn prox_ponto(&self, mut x: Ponto, passo: NumReal) -> Ponto {
        x[x.len() - 1] += passo;

        for idx in (0..DIM).rev() {
            if x[idx] > self.x_max[idx] {
                x[idx] = self.x_min[idx];

                if idx > 0 {
                    x[idx - 1] += passo;
                }
            }
        }

        return x;
    }

    // Função que verifica se as restrições passam nas qualificações extendidas de Mangassarian Fromovitz
    // Se alguma condição for quebrada, retorna, se não, continua analisando até ter visto todo o espaço
    // E então retorna que passa nas qualificações
    pub fn emfcq(&self) -> bool {
        // Tem que analisar todo o espaço, nesse caso limitado, então começa do menor ponto possivel
        let mut x = self.x_min;

        // O espaço vai ser analisado em passo discreto desse tamanho
        let passo = 0.1;

        loop {
            //println!("x: {:?}", x);

            // Calcula os gradientes das funções de restrições de igualdades em x
            let grads_hr = self
                .restricoes_igualdades
                .iter()
                .map(|restricao| auto_grad(x, restricao.clone()))
                .collect();

            // Caso alguma dupla de gradientes sejam dependentes linearmente, então a qualificação já foi quebrada
            if self.restricoes_igualdades.len() > 1 {
                if sao_linearmente_dependentes(&grads_hr) {
                    println!("Gradientes de h_r(x) são linearmente dependentes ");
                    return false;
                }
            }

            // Agora o que falta é procurar um z para esse x que satisfaça as outras condições

            let mut z = self.x_min; // Começa do menor possivel do espaço
            let mut existe_z = false; // Assumo que não existe um z, até encontrar um, ou não
            loop {
                //println!("x: {:?}, z: {:?}", x, z);

                // Se o produto interno entre algum gradiente e z for diferente (ou desconsideravel), procurar outro z que satisfaça
                for grad_hr in &grads_hr {
                    if !prox_o_suficiente_de_zero(produto_interno(grad_hr, &z)) {
                        z = self.prox_ponto(z, passo);
                        continue;
                    }
                }

                // Se chegou até aqui, então pra esse ser o z certo, basta que todos os produtos internos
                // entre os gradientes das funções de restrições de desigualdades que foram violadas, ou
                // que estão quase sendo violadas em x, e z sejam negativos. O que garente que existe uma
                // direção de descida para a restrição não ser violada ou se afasta da zona de fronteira
                // do violamento.
                let mut algum_gj_falha = false;
                for &gj in self.restricoes_desigualdades.iter() {
                    if gj(x) >= 0.0 {
                        // Só as funções que estão na fronteira ou que já foram violadas
                        let grad = auto_grad(x, gj);
                        if produto_interno(&grad, &z).is_sign_positive() {
                            // Se o produto interno é positivo, já falhou com a condição, então proximo passo
                            algum_gj_falha = true;
                            break;
                        }
                    }
                }

                // Se nenhuma condição falhou, então temos um z para o x
                if !algum_gj_falha {
                    existe_z = true;
                    break; // Não precisa procurar outro z
                }

                // Mesma verificação que x para saber se todo o espaço já foi analisado
                z = self.prox_ponto(z, passo);
                if prox_o_suficiente_de_zero(dist(self.x_min, z)) {
                    break;
                }
            }

            // Caso não exista um z, mesmo depois de visitar todo o espaço, então não passou
            // nas qualificações
            if !existe_z {
                println!("Não existe z pra x: {:?}", x);
                return false;
            }

            // Caso tenha passado, vai ser analisado para o proximo x, a menos que já tenha visitado todo o espaço
            x = self.prox_ponto(x, passo);
            if prox_o_suficiente_de_zero(dist(self.x_min, x)) {
                break;
            }
        }

        // Analisado todo o espaço, e tudo de acordo com as restrições, então passou nas qualificações
        true
    }
}

// Considerando isso como infinitesimal
const DBL_EPS: f64 = 1e-12;

// Dimensão do problema, R^DIM
const DIM: usize = 2;

// main né
fn main() {
    let p = Problema {
        funcao_objetivo: |x: Ponto| x[0] * x[1],
        restricoes_igualdades: vec![|x: Ponto| x[0] + x[1] - 4.0],
        restricoes_desigualdades: vec![
            |x: Ponto| (x[0] - 2.0).powi(2) + (x[1] - 2.0).powi(2) - 2.0,
            |x: Ponto| (x[0] - 2.5).powi(2) + (x[1] - 2.0).powi(2) - 2.0,
        ],
        x_min: [-4.0, -4.0],
        x_max: [4.0, 4.0],
    };

    let problema_emfcq = p.emfcq();
    if problema_emfcq {
        println!("EMFCQ? Sim");
    } else {
        println!("EMFCQ? Não :/");
        std::process::exit(1);
    }

    // Só pra evitar warnings
    let _x: Ponto = [3.0, 2.0];

    let _ = p.restricoes_igualdades;
    let _ = p.restricoes_desigualdades;
    let _ = p.x_min;
    let _ = p.x_max;
}
