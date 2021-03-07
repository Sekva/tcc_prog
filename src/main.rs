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

    //Armazenamento dos estados
    pub xs_linear: Vec<Ponto>,
    pub ds_linear: Vec<Ponto>,
}

impl Problema {
    // Retorna um novo problema preenchido a partir do minimo
    pub fn novo(
        funcao_objetivo: Funcao,
        restricoes_desigualdades: Vec<Funcao>,
        restricoes_igualdades: Vec<Funcao>,
    ) -> Problema {
        Problema {
            funcao_objetivo,
            restricoes_igualdades,
            restricoes_desigualdades,
            x_min: [-4.0, -4.0],
            x_max: [4.0, 4.0],
            xs_linear: Vec::new(),
            ds_linear: Vec::new(),
        }
    }

    // Resolve o subproblema linear por força bruta
    // Toma o ponto fixado xⁱ para calcular nele
    // Retorna um quintupla (valor minimo, d, tg, th⁺, th⁻)
    pub fn resolver_problema_linear_bruto(
        &self,
        x: Ponto,
    ) -> (
        NumReal,
        Vec<NumReal>,
        Vec<NumReal>,
        Vec<NumReal>,
        Vec<NumReal>,
    ) {
        let x_i = x;

        let grad_funcao_objetivo = auto_grad(x_i, self.funcao_objetivo);

        let grads_funcao_igualdades: Vec<Ponto> = self
            .restricoes_igualdades
            .iter()
            .map(|&f| auto_grad(x_i, f))
            .collect();

        let grads_funcao_desigualdades: Vec<Ponto> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| auto_grad(x_i, f))
            .collect();

        let funcao_desigualdades_avaliadas: Vec<NumReal> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| f(x_i))
            .collect();

        let funcao_igualdades_avaliadas: Vec<NumReal> =
            self.restricoes_igualdades.iter().map(|&f| f(x_i)).collect();

        // quantidade de variaveis
        let mi = self.restricoes_desigualdades.len(); //dimensão do vetor tg
        let me = self.restricoes_igualdades.len(); //dimensão dos vetores th+ e th-

        let passo = 0.1;
        let mut pontos_viaveis: Vec<(
            NumReal,
            Vec<NumReal>,
            Vec<NumReal>,
            Vec<NumReal>,
            Vec<NumReal>,
        )> = Vec::new();

        let mut dp = Vec::from(self.x_min);
        let d_min = Vec::from(self.x_min);
        let d_max = Vec::from(self.x_max);

        let mut tgp = vec![0.0; mi];
        let tg_min = vec![0.0; mi];
        let tg_max: Vec<NumReal> = funcao_desigualdades_avaliadas
            .iter()
            .map(|&v| {
                if v < 0.0 {
                    return 0.0;
                }
                return v;
            })
            .collect();

        let mut thpp = vec![0.0; me];
        let mut thmp = vec![0.0; me];
        let thx_min = vec![0.0; me];
        let thx_max = funcao_igualdades_avaliadas
            .iter()
            .map(|&v| v.abs())
            .collect();

        let mut todos_possiveis_ds = Vec::new();
        let mut todos_possiveis_tgs = Vec::new();
        let mut todos_possiveis_thps = Vec::new();
        let mut todos_possiveis_thms = Vec::new();

        loop {
            // Restrição (1d)
            todos_possiveis_ds.push(dp.clone());
            dp = prox_ponto(dp.clone(), &d_min, &d_max, DIM, passo);
            if iguais(&dp, &d_min, DIM) {
                break;
            }
        }
        loop {
            // Restrição (1e)
            todos_possiveis_tgs.push(tgp.clone());
            tgp = prox_ponto(tgp.clone(), &tg_min, &tg_max, mi, passo);
            if iguais(&tgp, &tg_min, mi) {
                break;
            }
        }
        loop {
            // Restrição (1f)
            todos_possiveis_thps.push(thpp.clone());
            thpp = prox_ponto(thpp.clone(), &thx_min, &thx_max, me, passo);
            if iguais(&thpp, &thx_min, me) {
                break;
            }
        }
        loop {
            // Restrição (1g)
            todos_possiveis_thms.push(thmp.clone());
            thmp = prox_ponto(thmp.clone(), &thx_min, &thx_max, me, passo);
            if iguais(&thmp, &thx_min, me) {
                break;
            }
        }

        for d in todos_possiveis_ds {
            let restricao_1c_falha = false;
            // TODO: Restrição (1c)

            // calculo do lado esquedo da expressão (1a)
            let mut lado_esq_1a = Vec::new();
            for idx in 0..mi {
                lado_esq_1a.push(
                    funcao_desigualdades_avaliadas[idx]
                        + produto_interno(
                            &grads_funcao_desigualdades[idx],
                            &vec_arr_fixo(d.clone()),
                        ),
                );
            }

            // calculo do lado esquedo da expressão (1b)
            let mut lado_esq_1b = Vec::new();
            for idx in 0..me {
                lado_esq_1b.push(
                    funcao_igualdades_avaliadas[idx]
                        + produto_interno(&grads_funcao_igualdades[idx], &vec_arr_fixo(d.clone())),
                );
            }

            if !restricao_1c_falha {
                for tg in &todos_possiveis_tgs {
                    let mut restricao_1a_falha = false;

                    for idx in 0..mi {
                        if !(lado_esq_1a[idx] <= tg[idx]) {
                            restricao_1a_falha = true;
                            break;
                        }
                    }

                    if !restricao_1a_falha {
                        for thp in &todos_possiveis_thps {
                            for thm in &todos_possiveis_thms {
                                let mut restricao_1b_falha = false;
                                for idx in 0..me {
                                    if !(prox_o_suficiente_de_zero(
                                        lado_esq_1b[idx] - (thp[idx] - thm[idx]),
                                    )) {
                                        restricao_1b_falha = true;
                                        break;
                                    }
                                }

                                if !restricao_1b_falha {
                                    let valor_funcao_objetivo: NumReal = produto_interno(
                                        &grad_funcao_objetivo,
                                        &vec_arr_fixo(d.clone()),
                                    ) + C
                                        * (tg.iter().sum::<NumReal>()
                                            + thp.iter().sum::<NumReal>()
                                            + thm.iter().sum::<NumReal>());

                                    pontos_viaveis.push((
                                        valor_funcao_objetivo,
                                        d.clone(),
                                        tg.clone(),
                                        thp.clone(),
                                        thm.clone(),
                                    ));
                                }
                            }
                        }
                    }
                }
            }
        }

        if pontos_viaveis.is_empty() {
            panic!("Não existe solução para o problema linear");
        }

        // procurando o menor em pontos viaveis
        let mut min_ponto_viavel = &pontos_viaveis[0];

        for ponto in &pontos_viaveis {
            if ponto.0 < min_ponto_viavel.0 {
                min_ponto_viavel = ponto;
            }
        }

        return min_ponto_viavel.clone();
    }

    // NÃO FUNCIONA
    // Função que resolve o problema linear
    // Retorna um unico vetor v, onde:
    // v_0..n é o vetor d
    // v_n..(n+mi) é o vetor tg
    // v_(n+mi)..(n+mi+me) é o vetor th+
    // v_(n+mi+me)..(n+mi+me+me) é o vetor th-
    pub fn resolver_problema_linear(&self, x: Ponto) -> Vec<NumReal> {
        // Biblioteca usada
        use good_lp::*;

        let x_i = x;

        let grad_funcao_objetivo = auto_grad(x_i, self.funcao_objetivo);

        let grads_funcao_igualdades: Vec<Ponto> = self
            .restricoes_igualdades
            .iter()
            .map(|&f| auto_grad(x_i, f))
            .collect();

        let grads_funcao_desigualdades: Vec<Ponto> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| auto_grad(x_i, f))
            .collect();

        let funcao_desigualdades_avaliadas: Vec<NumReal> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| f(x_i))
            .collect();

        let funcao_igualdades_avaliadas: Vec<NumReal> = self
            .restricoes_desigualdades
            .iter()
            .map(|&f| f(x_i))
            .collect();

        // quantidade de variaveis
        let d_max = DIM; //dimensão do vetor d
        let mi = self.restricoes_desigualdades.len(); //dimensão do vetor tg
        let me = self.restricoes_igualdades.len(); //dimensão dos vetores th+ e th-

        // gerenciador de variaveis
        let mut variaveis = variables!();

        // cria as listas de variaveis
        let mut d = Vec::new();
        let mut tg = Vec::new();
        let mut thp = Vec::new();
        let mut thm = Vec::new();

        // cria as variaveis d
        for _ in 0..d_max {
            d.push(variaveis.add(variable().min(f64::MIN).max(f64::MAX)));
        }

        // cria as variaveis tg
        for _ in 0..mi {
            tg.push(variaveis.add(variable().min(f64::MIN).max(f64::MAX)));
        }

        // cria as variaveis th+
        for _ in 0..me {
            thp.push(variaveis.add(variable().min(f64::MIN).max(f64::MAX)));
        }

        // cria as variaveis th-
        for _ in 0..me {
            thm.push(variaveis.add(variable().min(f64::MIN).max(f64::MAX)));
        }

        // Cria a função objetivo
        // ∇f(x_i)ᵀd + C·(soma(tg) + soma(th+) + soma(th-))

        // Começa a função pela segunda parte
        // C·(soma(tg) + soma(th+) + soma(th-))
        let mut funcao_objetivo_linear: Expression = C
            * (tg.iter().sum::<Expression>()
                + thp.iter().sum::<Expression>()
                + thm.iter().sum::<Expression>());

        // Adiciona as somas do produto interno ∇f(x_i)ᵀd isoladamente
        for idx in 0..d_max {
            funcao_objetivo_linear += grad_funcao_objetivo[idx] * d[idx];
        }

        // Cria a lista onde vão ficam as expreções de restrições
        let mut restricoes = Vec::new();

        // Restrição (1a)
        // g_j(x_i) + ∇g_j(x_i)ᵀd ≤ tg_j
        for j in 0..mi {
            let g_j_x_i: NumReal = funcao_desigualdades_avaliadas[j];
            let grad_g_j_x_i: Ponto = grads_funcao_desigualdades[j];
            let mut expressao_esq: Expression = Expression::from_other_affine(g_j_x_i);
            for idx in 0..DIM {
                expressao_esq += grad_g_j_x_i[idx] * d[idx];
            }
            restricoes.push(constraint!(expressao_esq <= tg[j]));
        }

        // Restrição (1b)
        // h_r(x_i) + ∇h_r(x_i)ᵀd == thp_r - thm_r
        for r in 0..me {
            let h_r_x_i: NumReal = funcao_igualdades_avaliadas[r];
            let grad_h_r_x_i: Ponto = grads_funcao_igualdades[r];
            let mut expressao_esq: Expression = Expression::from_other_affine(h_r_x_i);
            for idx in 0..DIM {
                expressao_esq += grad_h_r_x_i[idx] * d[idx];
            }
            restricoes.push(constraint!(expressao_esq == (thp[r] - thm[r])));
        }

        // TODO: Restrição (1c)

        // TODO: Restrição (1d)
        for idx in 0..DIM {
            restricoes.push(constraint!(self.x_min[idx] <= d[idx]));
            restricoes.push(constraint!(d[idx] <= self.x_max[idx]));
        }

        // Restrição (1e)
        let max = |a: NumReal, b: NumReal| {
            if a < b {
                return b;
            }
            return a;
        };

        for j in 0..mi {
            restricoes.push(constraint!(0 <= tg[j]));
            restricoes.push(constraint!(
                tg[j] <= max(0.0, funcao_desigualdades_avaliadas[j])
            ));
        }

        // Restrição (1f)
        for r in 0..me {
            restricoes.push(constraint!(0 <= thp[r]));
            restricoes.push(constraint!(thp[r] <= funcao_igualdades_avaliadas[r].abs()));
        }

        // Restrição (1g)
        for r in 0..me {
            restricoes.push(constraint!(0 <= thm[r]));
            restricoes.push(constraint!(thm[r] <= funcao_igualdades_avaliadas[r].abs()));
        }

        // Cria o problema
        let mut problema_linear_restrito = variaveis
            .minimise(funcao_objetivo_linear.clone())
            .using(default_solver);

        // Aplica as restrições no problema
        for restricao in restricoes {
            problema_linear_restrito = problema_linear_restrito.with(restricao);
        }

        // Busca as solução do problema
        let solucao = problema_linear_restrito.solve();

        match solucao {
            Ok(solucao) => {
                for (idx, &i) in d.iter().enumerate() {
                    println!("d[{}] = {:?}", idx, solucao.value(i));
                }
                for (idx, &i) in tg.iter().enumerate() {
                    println!("tg[{}] = {:?}", idx, solucao.value(i));
                }
                for (idx, &i) in thp.iter().enumerate() {
                    println!("thp[{}] = {:?}", idx, solucao.value(i));
                }

                for (idx, &i) in thm.iter().enumerate() {
                    println!("thm[{}] = {:?}", idx, solucao.value(i));
                }
            }
            Err(e) => println!("aa {:?}", e),
        }

        return Vec::new();
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
                        z = vec_arr_fixo(prox_ponto(
                            Vec::from(z),
                            &Vec::from(self.x_min),
                            &Vec::from(self.x_max),
                            DIM,
                            passo,
                        ));
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
                z = vec_arr_fixo(prox_ponto(
                    Vec::from(z),
                    &Vec::from(self.x_min),
                    &Vec::from(self.x_max),
                    DIM,
                    passo,
                ));
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
            x = vec_arr_fixo(prox_ponto(
                Vec::from(x),
                &Vec::from(self.x_min),
                &Vec::from(self.x_max),
                DIM,
                passo,
            ));
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

// Constante de inviabilidade das iterações lineares
const C: NumReal = 1.0;

// main né
fn main() {
    let funcao_objetivo: Funcao = |x: Ponto| x[0] * x[1];
    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0]];
    let restricoes_desigualdades: Vec<Funcao> = vec![
        |x: Ponto| (x[0]).powi(2) + (x[1]).powi(2) - 2.0,
        |x: Ponto| (x[0]).powi(2) + (x[1]).powi(2) - 4.0,
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
    );

    let problema_emfcq = true; //p.emfcq();
    if problema_emfcq {
        println!("EMFCQ? Sim");
    } else {
        println!("EMFCQ? Não :/");
        std::process::exit(1);
    }

    let x: Ponto = [1., 1.];

    let solucao = p.resolver_problema_linear(x);
    println!("{:?}", solucao);

    // Só pra evitar warnings
    let _ = p.restricoes_igualdades;
    let _ = p.restricoes_desigualdades;
    let _ = p.x_min;
    let _ = p.x_max;
}
