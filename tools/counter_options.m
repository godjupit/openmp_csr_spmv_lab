// Build the archived event entries expected by xctrace's CPU Counters JSON.
// Uses Xcode's own event serializer; requires full Xcode on this Mac.
#import <Foundation/Foundation.h>
#import <dlfcn.h>

int main(int argc, char **argv) {
    @autoreleasepool {
        const char *path = getenv("CSR_INSTRUMENTS_PLUGIN");
        if (!path || !dlopen(path, RTLD_LAZY | RTLD_GLOBAL)) {
            fprintf(stderr, "Cannot load Instruments event serializer\n");
            return 1;
        }
        Class cls = NSClassFromString(@"XRCountersSetupEventOrFormula");
        if (!cls) return 1;
        NSMutableArray *events = [NSMutableArray array];
        for (int i = 1; i < argc; ++i) {
            id event = [cls new];
            [event setValue:[NSString stringWithUTF8String:argv[i]] forKey:@"aliasOrMnemonic"];
            NSError *error = nil;
            NSData *data = [NSKeyedArchiver archivedDataWithRootObject:event requiringSecureCoding:NO error:&error];
            if (!data || error) return 1;
            [events addObject:[data base64EncodedStringWithOptions:0]];
        }
        NSData *json = [NSJSONSerialization dataWithJSONObject:events options:0 error:nil];
        puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
    }
}
