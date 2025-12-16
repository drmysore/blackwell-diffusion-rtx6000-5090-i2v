{
  perSystem =
    { config, ... }:
    {
      openapi.pkgs.fal-ai = config.openapi.mkDerivation {
        pname = "fal.ai.openapi";
        version = "1.0.0";
        srcs = [ ./schnell.fal.ai.openapi.json ];

        # Relax specific rules for this external API
        redoclyRules = {
          # "operation-description" = "off"; # They might not have descriptions
          # "info-license" = "off"; # External API might not specify license
          # "operation-4xx-response" = "off"; # They might not document error responses
          # "paths-kebab-case" = "off"; # They might use different naming
        };
      };
    };
}
