#import <Foundation/Foundation.h>
#import <PDFKit/PDFKit.h>
#import <AppKit/AppKit.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        if (argc != 3) return 1;
        PDFDocument *doc = [[PDFDocument alloc] initWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]]];
        NSString *out = [NSString stringWithUTF8String:argv[2]];
        [[NSFileManager defaultManager] createDirectoryAtPath:out withIntermediateDirectories:YES attributes:nil error:nil];
        if (!doc) return 1;
        NSMutableString *text = [NSMutableString stringWithFormat:@"Pages: %lu\n", (unsigned long)doc.pageCount];
        for (NSUInteger i = 0; i < doc.pageCount; ++i) {
            PDFPage *page = [doc pageAtIndex:i];
            [text appendFormat:@"\n--- PAGE %lu ---\n%@\n", (unsigned long)i + 1, page.string];
            NSImage *image = [page thumbnailOfSize:NSMakeSize(1200, 1700) forBox:kPDFDisplayBoxMediaBox];
            NSBitmapImageRep *bitmap = [NSBitmapImageRep imageRepWithData:image.TIFFRepresentation];
            NSData *data = [bitmap representationUsingType:NSBitmapImageFileTypePNG properties:@{}];
            [data writeToFile:[out stringByAppendingPathComponent:[NSString stringWithFormat:@"page_%lu.png", (unsigned long)i + 1]] atomically:YES];
        }
        [text writeToFile:[out stringByAppendingPathComponent:@"text.txt"] atomically:YES encoding:NSUTF8StringEncoding error:nil];
        printf("%lu pages rendered\n", (unsigned long)doc.pageCount);
    }
}
