#[derive(Debug, PartialEq, Eq, Hash, Clone, Copy)]
pub enum TokenType {
    // single-char tokens
    LeftParen, RightParen, Comma, LeftSquare, RightSquare, LeftBracket, RightBracket, Semi, Equals,

    // one-or-two char tokens
    Plus, Minus, Slash, Star, EqualsEquals, GreaterThan, LessThan,
    LessThanEquals, GreaterThanEquals,

    // swizzling
    Dot, X, Y, Z, W, R, G, B, A,

    // keywords
    If, Else, Repeat, Macro, Return, Give,

    // literals
    Number, Vector,

    // identifiers
    Identifier,

    Eof
}