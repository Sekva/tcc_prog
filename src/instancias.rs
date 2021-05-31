use crate::defs::{Funcao, Ponto, Problema, A, B, CC, D, DIM, E, F, L1, L2, O1, O2};

fn _problema_incial() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| (x[0].powi(2) - x[1].powi(2) - 1.0).sqrt();

    // f3: https://www.sfu.ca/~ssurjano/boha.html

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[1]];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        // Fechar a caixinha toda
        |x: Ponto| x[0] + x[1] - 15.0,
        |x: Ponto| x[0] - x[1] - 15.0,
        |x: Ponto| -x[0] + x[1] - 15.0,
        |x: Ponto| -x[0] - x[1] - 15.0,
        |x: Ponto| {
            A * (x[0] - L1).powi(2) + B * (x[1] - L2).powi(2) - CC * (x[0] - L1) * (x[1] - L2)
                + D * (x[0] - L1)
                + E * (x[1] - L2)
                + F
        },
        |x: Ponto| {
            A * (x[0] - O1).powi(2)
                + B * (x[1] - O2).powi(2)
                + CC * (x[0] - O1) * (x[1] - O2)
                + D * (x[0] - O1)
                + E * (x[1] - O2)
                + F
        },
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-4.0, -4.0],
        [4.0, 4.0],
        [1.0, 0.0],
        None,
        "Inicial".into(),
    );

    return p;
}

fn _problema_bohachevsky() -> Problema {
    // f3: https://www.sfu.ca/~ssurjano/boha.html
    let funcao_objetivo: Funcao = |x: Ponto| {
        x[0].powi(2) + 2.0 * x[1].powi(2)
            - 0.3 * (3.0 * 3.1415926 * x[0] + 4.0 * 3.1415926 * x[1]).cos()
            + 0.3
    };

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[1]];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        // Fechar a caixinha toda
        |x: Ponto| x[0] + x[1] - 15.0,
        |x: Ponto| x[0] - x[1] - 15.0,
        |x: Ponto| -x[0] + x[1] - 15.0,
        |x: Ponto| -x[0] - x[1] - 15.0,
        |x: Ponto| {
            A * (x[0] - L1).powi(2) + B * (x[1] - L2).powi(2) - CC * (x[0] - L1) * (x[1] - L2)
                + D * (x[0] - L1)
                + E * (x[1] - L2)
                + F
        },
        |x: Ponto| {
            A * (x[0] - O1).powi(2)
                + B * (x[1] - O2).powi(2)
                + CC * (x[0] - O1) * (x[1] - O2)
                + D * (x[0] - O1)
                + E * (x[1] - O2)
                + F
        },
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-4.0, -4.0],
        [4.0, 4.0],
        [1.0, 0.0],
        Some([0.0, 0.0]),
        "Bohachevsky f3".into(),
    );

    return p;
}

fn _problema_perm_function() -> Problema {
    // https://www.sfu.ca/~ssurjano/perm0db.html

    let funcao_objetivo: Funcao = |x: Ponto| {
        let beta = 2.0;
        let mut soma_1 = 0.0;
        for i in 1..(DIM + 1) {
            let mut val = 0.0;

            for j in 1..(DIM + 1) {
                val += (j as f64 + beta)
                    * (x[j - 1].powi(i as i32) - (1.0 / ((j as f64).powi(i as i32))));
            }

            soma_1 += val * val;
        }

        soma_1
    };

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0] - 1.0];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        |x: Ponto| x[0] - DIM as f64,
        |x: Ponto| x[1] - DIM as f64,
        |x: Ponto| x[0] + DIM as f64,
        |x: Ponto| x[1] + DIM as f64,
        |x: Ponto| {
            A * (x[0] - L1).powi(2) + B * (x[1] - L2).powi(2) - CC * (x[0] - L1) * (x[1] - L2)
                + D * (x[0] - L1)
                + E * (x[1] - L2)
                + F
        },
        |x: Ponto| {
            A * (x[0] - O1).powi(2)
                + B * (x[1] - O2).powi(2)
                + CC * (x[0] - O1) * (x[1] - O2)
                + D * (x[0] - O1)
                + E * (x[1] - O2)
                + F
        },
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-5.0 * DIM as f64, -5.0 * DIM as f64],
        [5.0 * DIM as f64, 5.0 * DIM as f64],
        [1.5, 1.5],
        Some([1.0, 0.5]),
        "Perm".into(),
    );

    return p;
}

fn _problema_trid_function() -> Problema {
    // https://www.sfu.ca/~ssurjano/trid.html

    let funcao_objetivo: Funcao = |x: Ponto| {
        let mut val_1 = 0.0;
        for i in 1..(DIM + 1) {
            val_1 += (x[i - 1] - 1.0).powi(2);
        }

        let mut val_2 = 0.0;
        for i in 2..(DIM + 1) {
            val_2 += x[i - 1] * x[i - 1 - 1];
        }

        val_1 - val_2
    };

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0] + x[1] - 4.0];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        |x: Ponto| x[0] - (DIM as f64).powi(DIM as i32),
        |x: Ponto| x[1] - (DIM as f64).powi(DIM as i32),
        |x: Ponto| x[0] + (DIM as f64).powi(DIM as i32),
        |x: Ponto| x[1] + (DIM as f64).powi(DIM as i32),
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-10.0, -10.0],
        [10.0, 10.0],
        [15.0, 7.5],
        Some([2.0, 2.0]),
        "Trid".into(),
    );

    return p;
}

fn _problema_sum_squares() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| {
        let mut soma = 0.0;

        for i in 1..(DIM + 1) {
            soma += (i as f64) * x[i - 1].powi(2);
        }

        return soma;
    };

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0]];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        |x: Ponto| x[0] - (DIM as f64).powi(DIM as i32),
        |x: Ponto| x[1] - (DIM as f64).powi(DIM as i32),
        |x: Ponto| x[0] + (DIM as f64).powi(DIM as i32),
        |x: Ponto| x[1] + (DIM as f64).powi(DIM as i32),
        |x: Ponto| {
            A * (x[0] - L1).powi(2) + B * (x[1] - L2).powi(2) - CC * (x[0] - L1) * (x[1] - L2)
                + D * (x[0] - L1)
                + E * (x[1] - L2)
                + F
        },
        |x: Ponto| {
            A * (x[0] - O1).powi(2)
                + B * (x[1] - O2).powi(2)
                + CC * (x[0] - O1) * (x[1] - O2)
                + D * (x[0] - O1)
                + E * (x[1] - O2)
                + F
        },
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-10.0, -10.0],
        [10.0, 10.0],
        [1.5, 1.5],
        Some([0.0, 0.0]),
        "Sum squares".into(),
    );

    return p;
}

fn _problema_217() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| -x[1];

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0] * x[0] + x[1] * x[1] - 1.0];

    let restricoes_desigualdades: Vec<Funcao> = vec![|x: Ponto| -(1.0 + x[0] - 2.0 * x[1])];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-100.0, -100.0],
        [100.0, 100.0],
        [10.0, 10.0],
        Some([0.6, 0.8]),
        "217".into(),
    );

    return p;
}

fn _problema_221() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| -x[0];

    let restricoes_igualdades: Vec<Funcao> = vec![];

    let restricoes_desigualdades: Vec<Funcao> = vec![|x: Ponto| -((1.0 - x[0]).powi(3) - x[1])];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-100.0, -100.0],
        [100.0, 100.0],
        [0.25, 0.25],
        Some([1.0, 0.0]),
        "221".into(),
    );

    return p;
}

fn _problema_313() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| (x[0] - 20.0).powi(2) + (x[1] + 20.0).powi(2);

    let restricoes_igualdades: Vec<Funcao> =
        vec![|x: Ponto| ((x[0].powi(2)) / 100.0) + ((x[1].powi(2)) / 36.0) - 1.0];

    let restricoes_desigualdades: Vec<Funcao> = vec![
        |x: Ponto| x[0] + x[1] - 15.0,
        |x: Ponto| x[0] - x[1] - 15.0,
        |x: Ponto| -x[0] + x[1] - 15.0,
        |x: Ponto| -x[0] - x[1] - 15.0,
    ];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-100.0, -100.0],
        [100.0, 100.0],
        [0.0, 0.0],
        Some([7.809, -3.748]),
        "313".into(),
    );

    return p;
}

fn _problema_325() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| x[0].powi(2) + x[1];

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0].powi(2) + x[1].powi(2) - 9.0];

    let restricoes_desigualdades: Vec<Funcao> =
        vec![|x: Ponto| -(-(x[0] + x[1]) + 1.0), |x: Ponto| {
            -(-(x[0] + x[1].powi(2)) + 1.0)
        }];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-100.0, -100.0],
        [100.0, 100.0],
        [-3.0, 0.0],
        Some([-2.732, -1.536]),
        "325".into(),
    );

    return p;
}

fn _problema_14() -> Problema {
    let funcao_objetivo: Funcao = |x: Ponto| (x[0] - 2.0).powi(2) + (x[1] - 1.0).powi(2);

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0] - 2.0 * x[1] + 1.0];

    let restricoes_desigualdades: Vec<Funcao> =
        vec![|x: Ponto| -((-0.25 * (x[0].powi(2))) - x[1].powi(2) + 1.0)];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-100.0, -100.0],
        [100.0, 100.0],
        [2.0, 2.0],
        Some([0.8228756555322954, 0.9114378277661477]),
        "14".into(),
    );

    return p;
}

fn _problema_1() -> Problema {
    let funcao_objetivo: Funcao =
        |x: Ponto| 100.0 * (x[1] - x[0].powi(2)).powi(2) + (1.0 - x[0]).powi(2);

    let restricoes_igualdades: Vec<Funcao> = vec![|x: Ponto| x[0] - 1.0];

    let restricoes_desigualdades: Vec<Funcao> = vec![|x: Ponto| -x[1] - 1.5];

    let p = Problema::novo(
        funcao_objetivo,
        restricoes_desigualdades,
        restricoes_igualdades,
        [-100.0, -100.0],
        [100.0, 100.0],
        [-2.0, 1.0],
        Some([1.0, 1.0]),
        "1".into(),
    );

    return p;
}

fn problemas_cuia() -> Vec<Problema> {
    vec![
        _problema_bohachevsky(), // FUNCIONA
        // _problema_perm_function(), // NÃO FUNCIONA??
        _problema_trid_function(), // NÃO FUNCIONA?
                                   // _problema_sum_squares(),   // NÃO FUNCIONA?
    ]
}

fn problemas_10() -> Vec<Problema> {
    vec![
        // _problema_1(),  // NÃO FUNCIONA?
        // _problema_14(), // NÃO FUNCIONA?? FUNCIONA SEM 1C
    ]
}

fn problemas_17() -> Vec<Problema> {
    vec![
        // _problema_217(), // FUNCIONA
        // _problema_221(), // NÃO FUNCIONA? NEM ENCONTRA SEM 1C
        // _problema_313(), // NÃO FUNCIONA??
        // _problema_325(), // NÃO FUNCIONA?? FUNCIONA MELHOR SEM 1C
    ]
}

pub fn gerar_instancias() -> Vec<Problema> {
    let mut lista: Vec<Problema> = Vec::new();

    for _p in problemas_cuia() {
        lista.push(_p);
    }
    for _p in problemas_10() {
        lista.push(_p);
    }
    for _p in problemas_17() {
        lista.push(_p);
    }

    lista
}
