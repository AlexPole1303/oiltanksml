import AppKit
import PDFKit
import Vision

for path in CommandLine.arguments.dropFirst() {
    let url = URL(fileURLWithPath: path)
    guard let document = PDFDocument(url: url) else { fatalError("Cannot read PDF") }
    var pages = [[String]]()
    for index in 0..<document.pageCount {
        guard let page = document.page(at: index) else { fatalError("Missing page") }
        let bounds = page.bounds(for: .mediaBox)
        let size = NSSize(width: bounds.width * 3, height: bounds.height * 3)
        let image = page.thumbnail(of: size, for: .mediaBox)
        var rect = NSRect(origin: .zero, size: size)
        guard let bitmap = image.cgImage(forProposedRect: &rect, context: nil, hints: nil) else {
            fatalError("Cannot rasterize page")
        }
        let recognition = VNRecognizeTextRequest()
        recognition.recognitionLevel = .accurate
        recognition.recognitionLanguages = ["en-US"]
        try VNImageRequestHandler(cgImage: bitmap).perform([recognition])
        pages.append(recognition.results?.compactMap { $0.topCandidates(1).first?.string } ?? [])
    }
    let data = try JSONEncoder().encode(pages)
    try data.write(to: URL(fileURLWithPath: "tmp").appendingPathComponent(url.deletingPathExtension().lastPathComponent + "_ocr.json"))
    print(url.lastPathComponent, pages.count, pages.map(\.count))
}
