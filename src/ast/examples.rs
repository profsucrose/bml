use super::ast::{Ast, BuiltIn, Env, Field, Op, Swizzle, Val};

use lasso::Rodeo;
use Ast::*;
fn b(a: Ast) -> Box<Ast> {
    Box::new(a)
}

pub fn closing_circle() -> (Rodeo, Env, Ast) {
    // macro dot[a, b] a.x * b.x + a.y * b.y
    // macro dist[a, b] sqrt(dot(a - b, a - b))

    let env = Env::default();

    let mut rodeo = Rodeo::new();
    let mut i = |ident: &str| rodeo.get_or_intern(ident);

    /*
        uv = coord / resolution
        return frag if dist(uv, [0.5, 0.5]) > frame / max_frame else [0.0; 4]
    */

    let ast = Block(vec![
        Assign(
            i("uv"),
            b(BinOp(
                b(Ident(i("coord"))),
                Op::Div,
                b(Ident(i("resolution"))),
            )),
        ),
        Return(b(If {
            cond: b(BinOp(
                b(Call(
                    BuiltIn::Dist,
                    vec![Ident(i("uv")), V(Val::Vec2(0.5, 0.5))],
                )),
                Op::More,
                b(BinOp(
                    b(Ident(i("frame"))),
                    Op::Div,
                    b(Ident(i("max_frame"))),
                )),
            )),
            true_ret: b(Ident(i("frag"))),
            false_ret: b(Ident(i("frag"))),
        })),
    ]);

    (rodeo, env, ast)
}

pub fn invert_color() -> (Rodeo, Env, Ast) {
    let env = Env::default();

    let mut rodeo = Rodeo::new();
    let mut i = |ident: &str| rodeo.get_or_intern(ident);

    let ast = Return(b(VecLiteral(
        b(BinOp(
            b(V(Val::Float(1.0))),
            Op::Sub,
            b(VecAccess(
                b(Ident(i("frag"))),
                Swizzle(Field::X, None, None, None),
            )),
        )),
        b(BinOp(
            b(V(Val::Float(1.0))),
            Op::Sub,
            b(VecAccess(
                b(Ident(i("frag"))),
                Swizzle(Field::Y, None, None, None),
            )),
        )),
        Some(b(BinOp(
            b(V(Val::Float(1.0))),
            Op::Sub,
            b(VecAccess(
                b(Ident(i("frag"))),
                Swizzle(Field::Z, None, None, None),
            )),
        ))),
        Some(b(V(Val::Float(1.0)))),
    )));

    (rodeo, env, ast)
}
