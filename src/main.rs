mod defs;
use defs::*;

mod utils;
use utils::*;

// Estrutura do problema
pub struct Problema {
    pub funcao_objetivo: Funcao, // A função objetivo que quer ser minimizada
    pub restricoes_igualdades: Vec<Funcao>, // A lista das m_e funções restrições de igualdades, onde h_r(x) == 0, para r = 1, ..., m_e
    pub restricoes_desigualdades: Vec<Funcao>, // A lista das m_i funções restrições de desigualdades, onde g_j(x) == 0, para j = 1, ..., m_i

    // dL e dU são pontos que limitam a buscam de direções d no subproblema linear
    // mas antes de ser usado no SCP, podem ser usados como pontos que restrigem o
    // espaço total do problema, e, no EMFCQ, procurar no R^n todo é impossivel,
    // além de que a implementação da verificação do EMFCQ exige que existam limites fixados
    pub d_l: Ponto,
    pub d_u: Ponto,

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
            d_l: [-4.0, -4.0],
            d_u: [4.0, 4.0],
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

        let mut dp = Vec::from(self.d_l);
        let d_min = Vec::from(self.d_l);
        let d_max = Vec::from(self.d_u);

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

    // NÃO TESTADO 100%
    // Função que resolve o subproblema linear avaliado em x_i
    // Retorna os 4 vetores em ordem, onde:
    // 1 - d
    // 2 - tg
    // 3 - th+
    // 4 - th-
    pub fn resolver_problema_linear(
        &self,
        x_i: Ponto,
    ) -> (Vec<NumReal>, Vec<NumReal>, Vec<NumReal>, Vec<NumReal>) {
        // Biblioteca usada
        use minilp::*;

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

        // Quantidade de variaveis
        let n = DIM; // Dimensão do vetor d
        let mi = self.restricoes_desigualdades.len(); // Dimensão do vetor tg
        let me = self.restricoes_igualdades.len(); // Dimensão dos vetores th+ e th-

        // Cria o problema de minimização
        let mut problema = Problem::new(OptimizationDirection::Minimize);

        // Cria o vetor c de coeficientes da função objetivo do subproblema linear
        let mut c: Vec<NumReal> = Vec::with_capacity(n + mi + 2 * me);

        // Os primeiros coeficientes (que são os coeficientes da direção d)
        // são os componenters do gradiente da função objetivo do subproblema linear
        for idx_componente_gradiente in 0..n {
            c.push(grad_funcao_objetivo[idx_componente_gradiente]);
        }

        // Depois são os coeficientes das variáveis t de ralaxamento
        // Os coeficientes são todos C
        for _ in n..(n + mi + 2 * me) {
            c.push(C);
        }

        // Cria o vetor d de variaveis que vão ser otimizadas
        let mut d: Vec<Variable> = Vec::with_capacity(n);

        // Cria o vetor tg de variaveis que vão ser otimizadas
        let mut tg: Vec<Variable> = Vec::with_capacity(mi);

        // Cria o vetor th+ de variaveis que vão ser otimizadas
        let mut thp: Vec<Variable> = Vec::with_capacity(me);

        // Cria o vetor th- de variaveis que vão ser otimizadas
        let mut thm: Vec<Variable> = Vec::with_capacity(me);

        // As restrições (1d), (1e), (1f) e (1g) podem ser vistas como descrições
        // do dominio das variaveis, então, como a biblioteca permite, já se pode
        // impor essas restrições como dominios mesmo na criação da variavel.

        // As variaveis d são restristas em seu dominio e tem como coeficientes
        // os componentes do gradiente da função objetivo principal avaliada em x_i.
        for componente_d in 0..n {
            let coeficiente = c[componente_d];
            // As restrições que marcam o dominio de d são dL e dU, como em (1d)
            let dominio = (self.d_l[componente_d], self.d_u[componente_d]);
            let var = problema.add_var(coeficiente, dominio);
            d.push(var);
        }

        // As variaveis tg são restristas em seu dominio
        // e tem como coeficientes o valor C.
        for j in 0..mi {
            let coeficiente = C;

            // Restrição (1e)
            let dominio_fechado = (0.0, max(0.0, funcao_desigualdades_avaliadas[j]));
            let var = problema.add_var(coeficiente, dominio_fechado);
            tg.push(var);
        }

        // As variaveis  th+ são restristas em seu dominio
        // e tem como coeficientes o valor C.
        for r in 0..me {
            let coeficiente = C;
            // Restrição (1f)
            let dominio_fechado = (0.0, funcao_igualdades_avaliadas[r].abs());
            let var = problema.add_var(coeficiente, dominio_fechado);
            thp.push(var);
        }

        // As variaveis  th- são restristas em seu dominio
        // e tem como coeficientes o valor C.
        for r in 0..me {
            let coeficiente = C;
            // Restrição (1g)
            let dominio_fechado = (0.0, funcao_igualdades_avaliadas[r].abs());
            let var = problema.add_var(coeficiente, dominio_fechado);
            thm.push(var);
        }

        // Por simplicidade, criar cada restrição separadamente

        // Restrição (1a) ajustada
        for j in 0..mi {
            // Cria um expressão linear vazia
            let mut expressao_linear = LinearExpr::empty();

            // Como é um produto interno, os componentes do gradiente de g_j já são
            // os coeficientes das variaveis que precisam ser encontradas
            for (idx_grad, &componente_d) in d.iter().enumerate() {
                expressao_linear.add(componente_d, grads_funcao_desigualdades[j][idx_grad]);
            }

            // Adiciona o ultimo termo que é tgj com coeficiente -1
            expressao_linear.add(tg[j], -1.0);
            problema.add_constraint(
                expressao_linear,
                ComparisonOp::Le,
                funcao_desigualdades_avaliadas[j],
            );
        }

        // Restrição (1b) ajustada
        for r in 0..me {
            // Cria um expressão linear vazia
            let mut expressao_linear = LinearExpr::empty();

            // Como é um produto interno, os componentes do gradiente de h_j já são
            // os coeficientes das variaveis que precisam ser encontradas
            for (idx_grad, &componente_d) in d.iter().enumerate() {
                expressao_linear.add(componente_d, grads_funcao_igualdades[r][idx_grad]);
            }

            // Adiciona os dois ultimos termo que são thp_r e thm_r com coeficientes -1 e 1
            expressao_linear.add(thp[r], -1.0);
            expressao_linear.add(thm[r], 1.0);
            problema.add_constraint(
                expressao_linear,
                ComparisonOp::Eq,
                -1.0 * funcao_igualdades_avaliadas[r],
            );
        }

        // TODO: Restrição (1c)

        // Finalmente pede pra resolver o problema
        let solucao = problema.solve();

        // Verifica a solução
        match solucao {
            // Se a solução foi um sucesso, ok, salva as informações da solução em s
            Ok(s) => {
                // Imprime na tela em caso de debug
                if DEBUG {
                    for (idx, &dc) in d.iter().enumerate() {
                        let strd = format!("d[{}] = {}", idx, s[dc]);
                        dbg!(strd);
                    }

                    for (idx, &tgc) in tg.iter().enumerate() {
                        let strd = format!("tg[{}] = {}", idx, s[tgc]);
                        dbg!(strd);
                    }

                    for (idx, &thpc) in thp.iter().enumerate() {
                        let strd = format!("thp[{}] = {}", idx, s[thpc]);
                        dbg!(strd);
                    }

                    for (idx, &thmc) in thm.iter().enumerate() {
                        let strd = format!("thm[{}] = {}", idx, s[thmc]);
                        dbg!(strd);
                    }
                }

                // Cria os vetores que vão ser retornados como solução
                let mut ds = Vec::new();
                let mut tgs = Vec::new();
                let mut thps = Vec::new();
                let mut thms = Vec::new();

                // Salva os valores de cada variavel otimizada
                for &dc in &d {
                    ds.push(s[dc]);
                }

                for &tgc in &tg {
                    tgs.push(s[tgc]);
                }

                for &thpc in &thp {
                    thps.push(s[thpc]);
                }

                for &thmc in &thm {
                    thms.push(s[thmc]);
                }

                // Por fim, finalmente, retorna
                return (ds, tgs, thps, thms);
            }

            // Caso a solução retorne um erro...
            _ => {
                // solucao.unwrap_err() retorna o tipo de erro e imprime na tela
                println!(
                    "Problema linear não tem solução ... {}",
                    solucao.unwrap_err()
                );

                // Encerra o programa, não tem mais sentido continuar
                std::process::exit(1);
            }
        }
    }

    // Função que verifica se as restrições passam nas qualificações extendidas de Mangassarian Fromovitz
    // Se alguma condição for quebrada, retorna, se não, continua analisando até ter visto todo o espaço
    // E então retorna que passa nas qualificações
    pub fn emfcq(&self) -> bool {
        // Tem que analisar todo o espaço, nesse caso limitado, então começa do menor ponto possivel
        let mut x = self.d_l;

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

            let mut z = self.d_l; // Começa do menor possivel do espaço
            let mut existe_z = false; // Assumo que não existe um z, até encontrar um, ou não
            loop {
                //println!("x: {:?}, z: {:?}", x, z);

                // Se o produto interno entre algum gradiente e z for diferente (ou desconsideravel), procurar outro z que satisfaça
                for grad_hr in &grads_hr {
                    if !prox_o_suficiente_de_zero(produto_interno(grad_hr, &z)) {
                        z = vec_arr_fixo(prox_ponto(
                            Vec::from(z),
                            &Vec::from(self.d_l),
                            &Vec::from(self.d_u),
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
                    &Vec::from(self.d_l),
                    &Vec::from(self.d_u),
                    DIM,
                    passo,
                ));
                if prox_o_suficiente_de_zero(dist(self.d_l, z)) {
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
                &Vec::from(self.d_l),
                &Vec::from(self.d_u),
                DIM,
                passo,
            ));
            if prox_o_suficiente_de_zero(dist(self.d_l, x)) {
                break;
            }
        }

        // Analisado todo o espaço, e tudo de acordo com as restrições, então passou nas qualificações
        true
    }
}

pub const DEBUG: bool = true;

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

    let x: Ponto = [1., 0.9];

    let solucao = p.resolver_problema_linear(x);
    println!("{:?}", solucao);

    // Só pra evitar warnings
    let _ = p.restricoes_igualdades;
    let _ = p.restricoes_desigualdades;
    let _ = p.d_l;
    let _ = p.d_u;
}
