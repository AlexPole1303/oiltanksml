import AppKit
import PDFKit
let arguments = CommandLine.arguments
let pdf = PDFDocument(url: URL(fileURLWithPath: arguments[1]))!
for value in arguments.dropFirst(2) {
    let page = pdf.page(at: Int(value)! - 1)!
    let bounds = page.bounds(for: .mediaBox)
    let image = page.thumbnail(of: NSSize(width: bounds.width*2, height: bounds.height*2), for: .mediaBox)
    let bits = NSBitmapImageRep(data: image.tiffRepresentation!)!
    try bits.representation(using: .png, properties: [:])!.write(to: URL(fileURLWithPath: "tmp/okeis_plains_\(value).png"))
}
