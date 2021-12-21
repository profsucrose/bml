use super::*;

use Ast::*;
fn b(a: Ast) -> Box<Ast> { Box::new(a) }

pub fn closing_circle() -> Ast {
  // macro dot[a, b] a.x * b.x + a.y * b.y 
  // macro dist[a, b] sqrt(dot(a - b, a - b))

  Block(vec![
    Assign(0, b(BinOp(b(Ident(3)), Op::Div, b(Ident(2))))),
    Return(b(If {
      cond: b(BinOp(
        b(Call(BuiltIn::Dist, vec![Ident(0), V(Val::Vec2(0.5, 0.5))])),
        Op::Less,
        b(BinOp(b(V(Val::Float(1.0))), Op::Sub, b(BinOp(b(Ident(6)), Op::Div, b(Ident(5))))))
      )),
      true_ret: b(Ident(4)),
      false_ret: b(V(Val::Vec4(0.0, 0.0, 0.0, 1.0))),
    }))
  ])
}

pub fn invert_color() -> Ast {
  Return(b(VecLiteral(
    b(BinOp(
      b(V(Val::Float(1.0))),
      Op::Sub,
      b(VecAccess(b(Ident(4)), Swizzle(Field::X, None, None, None))),
    )),
    b(BinOp(
      b(V(Val::Float(1.0))),
      Op::Sub,
      b(VecAccess(b(Ident(4)), Swizzle(Field::Y, None, None, None))),
    )),
    Some(b(BinOp(
      b(V(Val::Float(1.0))),
      Op::Sub,
      b(VecAccess(b(Ident(4)), Swizzle(Field::Z, None, None, None))),
    ))),
    Some(b(V(Val::Float(1.0)))),
  )))
}
