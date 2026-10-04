import CoreText
import Foundation

extension NSAttributedString.Key {
    static let ctFont = NSAttributedString.Key(kCTFontAttributeName as String)
    static let ctForegroundColor = NSAttributedString.Key(kCTForegroundColorAttributeName as String)
}

guard CommandLine.arguments.count == 3 else {
    fputs("usage: render_pdf.swift input.txt output.pdf\n", stderr)
    exit(2)
}

let input = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let text = try String(contentsOf: input, encoding: .utf8) as NSString
let pageWidth: CGFloat = 595.28
let pageHeight: CGFloat = 841.89
let margin: CGFloat = 48
let footerHeight: CGFloat = 24
let font = CTFontCreateWithName("Times New Roman" as CFString, 9.4, nil)
let attributes: [NSAttributedString.Key: Any] = [.ctFont: font]
let attributed = NSAttributedString(string: text as String, attributes: attributes)
let framesetter = CTFramesetterCreateWithAttributedString(attributed)
var mediaBox = CGRect(x: 0, y: 0, width: pageWidth, height: pageHeight)
guard let context = CGContext(output as CFURL, mediaBox: &mediaBox, nil) else {
    fputs("cannot create PDF context\n", stderr)
    exit(1)
}

let path = CGMutablePath()
path.addRect(CGRect(
    x: margin,
    y: margin + footerHeight,
    width: pageWidth - 2 * margin,
    height: pageHeight - 2 * margin - footerHeight
))

var location = 0
var pageNumber = 1
while location < text.length {
    context.beginPDFPage(nil)
    let frame = CTFramesetterCreateFrame(framesetter, CFRange(location: location, length: 0), path, nil)
    CTFrameDraw(frame, context)
    let visible = CTFrameGetVisibleStringRange(frame)
    if visible.length == 0 { break }
    location += visible.length
    let footer = NSAttributedString(
        string: "Morse-LiDAR · математика и методология · \(pageNumber)",
        attributes: [.ctFont: CTFontCreateWithName("Helvetica" as CFString, 7.5, nil), .ctForegroundColor: CGColor(gray: 0.35, alpha: 1)]
    )
    let footerLine = CTLineCreateWithAttributedString(footer)
    context.textPosition = CGPoint(x: margin, y: margin - 4)
    CTLineDraw(footerLine, context)
    context.endPDFPage()
    pageNumber += 1
}
context.closePDF()