#[derive(Debug)]
pub enum TokenType {
    // single-char tokens
    LeftParen, RightParen, Comma, LeftSquare, RightSquare, Semi, LeftBracket, RightBracket,

    // one-or-two char tokens
    Plus, Minus, Slash, Star, Equals, GreaterThan, LessThan,
    LessThanEquals, GreaterThanEquals,

    // swizzling
    Dot, X, Y, Z, W, R, G, B, A,

    // keywords
    If, Else, Repeat, Macro, Return, Give,

    // literals
    Number, Vector,

    Eof
}